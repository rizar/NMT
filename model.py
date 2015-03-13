# This is the original encoder-decoder model
# It only works with Blocks PR #414 merged. It seems to train, but
# I haven't sampled sentences yet
# TIP: Without CuDNN Theano seems to move part of the step clipping to CPU
#      on my computer, which makes things very slow. CuDNN gives a 2x speedup
#      in my case, so it's worth installing.
from collections import Counter
import numpy
import theano
from theano import tensor
from toolz import merge

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
                           Bias, Initializable)
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter, SequenceGenerator
)

from stream import masked_stream


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class GatedRecurrentWithContext(Initializable):
    def __init__(self, *args, **kwargs):
        self.gated_recurrent = GatedRecurrent(*args, **kwargs)
        self.children = [self.gated_recurrent]

    @application(states=['states'], outputs=['states'],
                 contexts=['readout_context', 'transition_context',
                           'update_context', 'reset_context'])
    def apply(self, transition_context, update_context, reset_context,
              *args, **kwargs):
        kwargs['inputs'] += transition_context
        kwargs['update_inputs'] += update_context
        kwargs['reset_inputs'] += reset_context
        # readout_context was only added for the Readout brick, discard it
        kwargs.pop('readout_context')
        return self.gated_recurrent.apply(*args, **kwargs)

    def get_dim(self, name):
        if name in ['readout_context', 'transition_context',
                    'update_context', 'reset_context']:
            return self.dim
        return self.gated_recurrent.get_dim(name)

    def __getattr__(self, name):
        if name == 'gated_recurrent':
            raise AttributeError
        return getattr(self.gated_recurrent, name)

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inputs']
        if self.use_update_gate:
            sequences.append('update_inputs')
        if self.use_reset_gate:
            sequences.append('reset_inputs')
        return sequences


class Encoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim, reverse=True,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.reverse = reverse

        self.lookup = LookupTable(name='embeddings')
        self.transition = GatedRecurrent(Tanh(), name='encoder_transition')
        self.fork = Fork([name for name in self.transition.apply.sequences
                          if name != 'mask'], prototype=Linear())

        self.children = [self.lookup, self.transition, self.fork]

    def _push_allocation_config(self):
        self.lookup.length = self.vocab_size
        self.lookup.dim = self.embedding_dim
        self.transition.dim = self.state_dim
        self.fork.input_dim = self.embedding_dim
        self.fork.output_dims = [self.state_dim
                                 for _ in self.fork.output_names]

    @application(inputs=['source_sentence', 'source_sentence_mask'],
                 outputs=['representation'])
    def apply(self, source_sentence, source_sentence_mask):
        # Time as first dimension
        source_sentence = source_sentence.dimshuffle(1, 0)
        if self.reverse:
            source_sentence = source_sentence[::-1]
            source_sentence_mask = source_sentence_mask.T[::-1]

        embeddings = self.lookup.apply(source_sentence)
        representation = self.transition.apply(**merge(
            self.fork.apply(embeddings, as_dict=True),
            {'mask': source_sentence_mask}
        ))
        return representation[-1]


class Decoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim

        readout = Readout(
            source_names=['states', 'feedback', 'readout_context'],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(),
            feedback_brick=LookupFeedback(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=1000).apply,
                 Maxout(num_pieces=2).apply,
                 Linear(input_dim=state_dim / 2, output_dim=100,
                        use_bias=False).apply,
                 Linear(input_dim=100).apply]),
            merged_dim=1000)

        self.transition = GatedRecurrentWithContext(Tanh(), dim=state_dim,
                                                    name='decoder')
        # Readout will apply the linear transformation to 'readout_context'
        # with a Merge brick, so no need to fork it here
        self.fork = Fork([name for name in
                          self.transition.apply.contexts +
                          self.transition.apply.states
                          if name != 'readout_context'], prototype=Linear())

        self.sequence_generator = SequenceGenerator(
            readout=readout, transition=self.transition,
            fork_inputs=[name for name in self.transition.apply.sequences
                         if name != 'mask'],
        )

        self.children = [self.fork, self.sequence_generator]

    def _push_allocation_config(self):
        self.fork.input_dim = self.representation_dim
        self.fork.output_dims = [self.state_dim
                                 for _ in self.fork.output_names]

    @application(inputs=['representation', 'target_sentence_mask',
                         'target_sentence'], outputs=['cost'])
    def cost(self, representation, target_sentence, target_sentence_mask):
        target_sentence = target_sentence.dimshuffle(1, 0)
        target_sentence_mask = target_sentence_mask.T

        # The initial state and contexts, all functions of the representation
        contexts = {key: value.dimshuffle('x', 0, 1)
                    if key not in self.transition.apply.states else value
                    for key, value
                    in self.fork.apply(representation, as_dict=True).items()}
        cost = self.sequence_generator.cost(**merge(
            contexts, {'mask': target_sentence_mask,
                       'outputs': target_sentence,
                       'readout_context': representation.dimshuffle('x', 0, 1)}
        ))

        return (cost * target_sentence_mask).sum() / target_sentence_mask.sum()


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
    encoder = Encoder(30000, 100, 1000)
    decoder = Decoder(30000, 100, 1000, 1000)
    cost = decoder.cost(encoder.apply(source_sentence, source_sentence_mask),
                        target_sentence, target_sentence_mask)

    # Initialize model
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(0.1)
    encoder.biases_init = decoder.biases_init = Constant(0)
    encoder.push_initialization_config()
    decoder.push_initialization_config()
    encoder.transition.weights_init = Orthogonal()
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
