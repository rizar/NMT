# This is the RNNsearch model
# Works with Blocks commit
# d9fcdec9f03bff762d9c9f85273dff8ed50a2b19
from collections import Counter
import argparse
import numpy
import theano
from theano import tensor
from toolz import merge
from picklable_itertools.extras import equizip

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.plot import Plot

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention, AttentionRecurrent
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork, Distribute
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    BaseSequenceGenerator, FakeAttentionRecurrent
)

from stream import masked_stream

# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass

class SequenceGenerator(BaseSequenceGenerator):
    """A more user-friendly interface for BaseSequenceGenerator.

    Parameters
    ----------
    readout : instance of :class:`AbstractReadout`
        The readout component for the sequence generator.
    transition : instance of :class:`.BaseRecurrent`
        The recurrent transition to be used in the sequence generator.
        Will be combined with `attention`, if that one is given.
    attention : :class:`.Brick`
        The attention mechanism to be added to ``transition``. Can be
        ``None``, in which case no attention mechanism is used.
    add_contexts : bool
        If ``True``, the :class:`AttentionRecurrent` wrapping the
        `transition` will add additional contexts for the attended and
        its mask.

    Notes
    -----
    Currently works only with lazy initialization (uses blocks that can not
    be constructed with a single call).

    """
    def __init__(self, readout, transition, attention=None,
                 fork_inputs=None, add_contexts=True, prototype=None, **kwargs):
        if not fork_inputs:
            fork_inputs = [name for name in transition.apply.sequences
                           if name != 'mask']

        fork = Fork(fork_inputs, prototype=prototype)
        if attention:
            distribute = Distribute(fork_inputs,
                                    attention.take_glimpses.outputs[0])
            transition = AttentionRecurrent(
                transition, attention, distribute,
                add_contexts=add_contexts, name="att_trans")
        else:
            transition = FakeAttentionRecurrent(transition,
                                                name="with_fake_attention")
        super(SequenceGenerator, self).__init__(
            readout, transition, fork, **kwargs)

class LookupFeedbackWMT15(LookupFeedback):

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()

        lookup_flat = tensor.switch(outputs_flat[:, None] < 0,
                      tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
                      self.lookup.apply(outputs_flat))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup

class SoftmaxEmitterWMT15(SoftmaxEmitter):

    @application
    def initial_outputs(self, batch_size, *args, **kwargs):
        return -tensor.ones((batch_size,), dtype='int64')

class BidirectionalWMT15(Bidirectional):

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]

class BidirectionalEncoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim

        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(GatedRecurrent(activation=Tanh(), dim=state_dim))
        self.fwd_fork = Fork([name for name in self.bidir.prototype.apply.sequences
                          if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork([name for name in self.bidir.prototype.apply.sequences
                          if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir, self.fwd_fork, self.back_fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim

        self.fwd_fork.input_dim = self.embedding_dim
        self.fwd_fork.output_dims = [self.state_dim
                                 for _ in self.fwd_fork.output_names]
        self.back_fork.input_dim = self.embedding_dim
        self.back_fork.output_dims = [self.state_dim
                                 for _ in self.back_fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.dimshuffle(1, 0)
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
            {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
            {'mask': source_sentence_mask})
        )
        return representation

class Decoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim

        self.transition = GatedRecurrent(dim=state_dim, activation=Tanh(), name='decoder')
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        readout = Readout(
            source_names=['states', 'feedback', self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitterWMT15(),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim).apply,
                 Maxout(num_pieces=2).apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False).apply,
                 Linear(input_dim=embedding_dim).apply]),
            merged_dim=state_dim)

        self.state_init = MLP(activations=[Tanh()], dims=[state_dim, state_dim],
                              name='state_initializer')

        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork_inputs=[name for name in self.transition.apply.sequences
                         if name != 'mask'],
            prototype=Linear()
        )

        self.children = [self.sequence_generator, self.state_init]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.dimshuffle(1, 0)
        target_sentence_mask = target_sentence_mask.T

        init_states = self.state_init.apply(representation[0, :, -self.state_dim:])

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(
                    **{'states':init_states,
                       'mask': target_sentence_mask,
                       'outputs': target_sentence,
                       'attended': representation,
                       'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum()

if __name__ == "__main__":

    # Create Theano variables
    source_sentence = tensor.lmatrix('english')
    source_sentence_mask = tensor.matrix('english_mask')
    target_sentence = tensor.lmatrix('french')
    target_sentence_mask = tensor.matrix('french_mask')

    # Test values
    theano.config.compute_test_value = 'warn'
    source_sentence.tag.test_value = numpy.random.randint(10, size=(10, 10))
    target_sentence.tag.test_value = numpy.random.randint(10, size=(10, 10))
    source_sentence_mask.tag.test_value = \
        numpy.random.rand(10, 10).astype('float32')
    target_sentence_mask.tag.test_value = \
        numpy.random.rand(10, 10).astype('float32')

    # Construct model
    encoder = BidirectionalEncoder(30000, 100, 1000)
    decoder = Decoder(30000, 100, 1000, 1000)
    cost = decoder.cost(encoder.apply(source_sentence, source_sentence_mask),
                        target_sentence, target_sentence_mask)

    # Initialize model
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(0.1)
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.bidir.prototype.weights_init = Orthogonal()
    decoder.transition.weights_init = Orthogonal()
    encoder.initialize()
    decoder.initialize()

    cg = ComputationGraph(cost)

    # Print shapes
    shapes = [param.get_value().shape for param in cg.parameters]
    print('Parameter shapes')
    for shape, count in Counter(shapes).most_common():
        print('    {:15}: {}'.format(shape, count))

    # Set up training algorithm
    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=CompositeRule([StepClipping(10), AdaDelta()])
    )

    # Train!
    main_loop = MainLoop(
        model=Model(cost),
        algorithm=algorithm,
        data_stream=masked_stream,
        extensions=[
            TrainingDataMonitoring([cost], after_every_batch=True),
            Plot('En-Fr', channels=[['decoder_cost_cost']],
                 after_every_batch=True),
            Printing(after_every_batch=True),
            Checkpoint('model.pkl', every_n_batches=2048)
        ]
    )
    main_loop.run()
