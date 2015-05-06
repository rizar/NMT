# This is the RNNsearch model
# Works with https://github.com/orhanf/blocks/tree/wmt15
# 0e23b0193f64dc3e56da18605d53d6f5b1352848
import argparse
import logging
import pprint
import theano
from theano import tensor
from theano.ifelse import ifelse
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
from blocks.extensions.saveload import Dump

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import (ShallowEnergyComputer,
                                     AbstractAttentionRecurrent,
                                     GenericSequenceAttention)
from blocks.bricks.base import application, lazy
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork, Parallel, Distribute
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional, recurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    BaseSequenceGenerator
)
from blocks.utils import dict_union, dict_subset

import config
import stream_fide_en

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_wmt15_fide_en_TEST",
                    help="Prototype config to use for config")
args = parser.parse_args()

# Make config global, nasty workaround since parameterizing stream
# will cause erroneous picklable behaviour, find a better solution
config = getattr(config, args.proto)()

# dictionary mapping stream name to stream getters
streams = {'fide-en': stream_fide_en}


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class MultiEncoder(Initializable):

    def __init__(self, config, **kwargs):
        super(MultiEncoder, self).__init__(**kwargs)

        self.num_encs = config['num_encs']
        self.encoders = []
        for i in xrange(self.num_encs):
            self.encoders.append(
                BidirectionalEncoder(
                    config['src_vocab_size_%d' % i],
                    config['enc_embed_%d' % i],
                    config['enc_nhids_%d' % i]))

        # this is the embedding from h to z
        self.annotation_embedders = [Linear(input_dim=(2 * config['enc_nhids_%d' % i]),
                                            output_dim=config['state_dim'],
                                            name='annotation_embedder_%d' % i,
                                            use_bias=False)
                                     for i in xrange(self.num_encs)]
        self.src_selector_embedder = Linear(input_dim=config['num_encs'],
                                            output_dim=config['state_dim'],
                                            use_bias=False,
                                            name='src_selector_embedder')
        self.trg_selector_embedder = Linear(input_dim=config['num_decs'],
                                            output_dim=config['state_dim'],
                                            use_bias=False,
                                            name='trg_selector_embedder')
        self.children = self.encoders + self.annotation_embedders +\
            [self.src_selector_embedder, self.trg_selector_embedder]

    @application
    def apply(self, source_sentences, source_masks, src_selector, trg_selector):

        # Projected Annotations
        rep = ifelse(
                theano.tensor.eq(src_selector[0], 1.),
                self.annotation_embedders[0].apply(
                    self.encoders[0].apply(source_sentences[0], source_masks[0])),
                self.annotation_embedders[1].apply(
                 self.encoders[1].apply(source_sentences[1], source_masks[1]))
        )

        # Source mask
        mask = ifelse(
                theano.tensor.eq(src_selector[0], 1.),
                source_masks[0], source_masks[1])

        # Source selector annotations, expand it to have batch size
        # dimensions for further ease in recurrence
        src_selector_rep = self.src_selector_embedder.apply(
                #theano.tensor.repeat(
                theano.tensor.repeat(
                    src_selector[None, :], rep.shape[1], axis=0)[None, :, :]
                #,rep.shape[0], axis=0)
        )
        # Target selector annotations, expand it similarly
        trg_selector_rep = self.trg_selector_embedder.apply(
                #theano.tensor.repeat(
                theano.tensor.repeat(
                    trg_selector[None, :], rep.shape[1], axis=0)[None, :, :]
                #,rep.shape[0], axis=0)
        )
        return rep, mask, src_selector_rep, trg_selector_rep


class LookupFeedbackWMT15(LookupFeedback):

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(outputs_flat[:, None] < 0,
                      tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
                      self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


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
        source_sentence = source_sentence.T
        source_sentence_mask = source_sentence_mask.T

        embeddings = self.lookup.apply(source_sentence)

        representation = self.bidir.apply(
            merge(self.fwd_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask}),
            merge(self.back_fork.apply(embeddings, as_dict=True),
                  {'mask': source_sentence_mask})
        )
        return representation


class GRUInitialState(GatedRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(GRUInitialState, self).__init__(**kwargs)
        self.attended_dim = attended_dim
        self.initial_transformer = MLP(activations=[Tanh()],
                                       dims=[attended_dim, self.dim],
                                       name='state_initializer')
        self.children.append(self.initial_transformer)

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        attended = kwargs['attended']
        if state_name == 'states':
            initial_state = self.initial_transformer.apply(
                attended[0, :, -self.attended_dim:])
            return initial_state
        dim = self.get_dim(state_name)
        if dim == 0:
            return tensor.zeros((batch_size,))
        return tensor.zeros((batch_size, dim))


class SequenceMultiContentAttention(GenericSequenceAttention, Initializable):
    """Should extend SequenceContentAttention"""

    @lazy(allocation=['match_dim'])
    def __init__(self, match_dim, attended_dims, state_transformer=None,
                 attended_transformers=None, energy_computer=None, **kwargs):

        # TODO: This is ugly, fix it
        kwargs['attended_dim'] = attended_dims[0]
        super(SequenceMultiContentAttention, self).__init__(**kwargs)
        self.match_dim = match_dim
        self.attended_dims = attended_dims
        self.state_transformer = state_transformer
        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        self.num_attended = len(attended_dims)
        if not attended_transformers:
            attended_transformers = [Linear(name="preprocess_%d" % i)
                                     for i in xrange(self.num_attended)]
        self.attended_transformers = attended_transformers

        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, energy_computer] +\
            attended_transformers

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        for i in xrange(self.num_attended):
            self.attended_transformers[i].input_dim = self.attended_dims[i]
            self.attended_transformers[i].output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application
    def compute_energies(self, attendeds, preprocessed_attendeds,
                         states):
        if not preprocessed_attendeds:
            preprocessed_attendeds = self.preprocess(attendeds)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)

        # Broadcasting of transformed states should be done automatically
        match_vectors = transformed_states.values()
        for att in preprocessed_attendeds:
            match_vectors += att
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)
        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attendeds, preprocessed_attendeds=None,
                      attended_mask=None, **states):
        energies = self.compute_energies(attendeds, preprocessed_attendeds,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(
                weights, attendeds[0])
        return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (['attended_%d' % i
                 for i in xrange(self.num_attended)] +\
                ['preprocessed_attended_%d' % i
                 for i in xrange(self.num_attended)] +\
                ['attended_mask'] + self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, attended):
        if name == "weighted_averages":
            return tensor.zeros((batch_size, self.attended_dims[0]))
        elif name == "weights":
            return tensor.zeros((batch_size, attended[0].shape[0]))
        raise ValueError("Unknown glimpse name {}".format(name))

    @application(inputs=['attended'],
                 outputs=['preprocessed_attended_0',
                          'preprocessed_attended_1',
                          'preprocessed_attended_2'])
    def preprocess(self, attended):
        preprocessed_attended = []
        for i, att in enumerate(attended):
            preprocessed_attended.append(
                self.attended_transformers[i].apply(att))
        return preprocessed_attended

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dims[0]
        if name in ['weights']:
            return 0
        if name in ['attended_%d' % i
                    for i in xrange(self.num_attended)]:
            return self.attended_dims[int(name[-1])]
        if name in ['preprocessed_attended_%d' % i
                    for i in xrange(self.num_attended)]:
            return self.match_dim
        import ipdb;ipdb.set_trace()
        return super(SequenceMultiContentAttention, self).get_dim(name)


class AttentionRecurrentWithMultiContext(AbstractAttentionRecurrent, Initializable):

    def __init__(self, num_contexts, transition, attention, **kwargs):
        super(AttentionRecurrentWithMultiContext, self).__init__(**kwargs)
        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)
        self._context_names = list(transition.apply.contexts)

        # This part is tricky
        self.num_contexts = num_contexts
        attended_names = ['attended_%d' % i for i in xrange(num_contexts)]
        attended_mask_name = 'attended_mask'
        self._context_names += attended_names + [attended_mask_name]
        normal_inputs = [name for name in self._sequence_names
                         if 'mask' not in name]
        distribute = Distribute(normal_inputs,
                                attention.take_glimpses.outputs[0])

        self.transition = transition
        self.attention = attention
        self.distribute = distribute
        self.attended_names = attended_names
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_names = ["preprocessed_" + attended_names[i]
                                            for i in xrange(num_contexts)]

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        self.children = [self.transition, self.attention, self.distribute]

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(
            self.attention.state_names)

        # TODO: this already pushed, check it
        #self.attention.attended_dims = self.get_dim(self.attended_name)

        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)
        result = self.attention.take_glimpses(
            [kwargs.pop(name) for name in self.attended_names],
            [kwargs.pop(name, None) for name in
                self.preprocessed_attended_names],
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed))
        if kwargs:
            raise ValueError("extra args to take_glimpses: {}".format(kwargs))
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @application
    def compute_states(self, **kwargs):
        # TODO: check this
        import ipdb;ipdb.set_trace()
        # Masks are not mandatory, that's why 'must_have=False'
        sequences = dict_subset(kwargs, self._sequence_names,
                                pop=True, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        for name in self.attended_names:
            kwargs.pop(name)
        kwargs.pop(self.attended_mask_name)

        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))
        # TODO: gru state conditioning for source selector should be done here
        current_states = self.transition.apply(
            iterate=False, as_list=True,
            **dict_union(sequences, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self._state_names

    @recurrent
    def do_apply(self, **kwargs):
        attendeds_dict = {}
        preprocessed_attendeds_dict = {}
        # ordering is important
        for i in xrange(self.num_contexts):
            att_name = self.attended_names[i]
            p_att_name = self.preprocessed_attended_names[i]
            attendeds_dict[att_name] = kwargs[att_name]
            preprocessed_attendeds_dict[p_att_name] = kwargs.pop(p_att_name)

        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

        current_glimpses = self.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses, attendeds_dict,
                preprocessed_attendeds_dict,
                {self.attended_mask_name: attended_mask}))
        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + self.preprocessed_attended_names

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names

    @application
    def apply(self, **kwargs):
        preprocessed_attended = self.attention.preprocess(
            [kwargs[name] for name in self.attended_names])
        add_kwargs = dict(zip(self.preprocessed_attended_names,
                              preprocessed_attended))
        new_kwargs = dict_union(kwargs, add_kwargs)
        return self.do_apply(**new_kwargs)

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    def get_dim(self, name):
        if name in self._glimpse_names:
            return self.attention.get_dim(name)
        if name in self.preprocessed_attended_names:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        # TODO: this is a bit tricky, find a better soln
        if name in self.attended_names:
            return self.attention.get_dim(
                self.attention.take_glimpses.inputs[int(name[-1])])
        if name == self.attended_mask_name:
            return 0
        return self.transition.get_dim(name)


class SequenceGeneratorWithMultiContext(BaseSequenceGenerator):
    def __init__(self, num_contexts, readout, transition, attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        self.num_contexts = num_contexts
        transition = AttentionRecurrentWithMultiContext(
            num_contexts, transition, attention,
            name="att_trans")
        super(SequenceGeneratorWithMultiContext, self).__init__(
            readout, transition, **kwargs)


class ReadoutWithMultiEncoder(Readout):

    def get_dim(self, name):
        # TODO: if we change the name to 'attended_1' then we wont need this class
        if name == 'src_selector_rep':
            return self.merged_dim
        return super(ReadoutWithMultiEncoder, self).get_dim(name)


class Decoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, src_selector_rep, trg_selector_rep,
                 num_encs, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.src_selector_rep = src_selector_rep
        self.trg_selector_rep = trg_selector_rep
        self.num_encs = num_encs

        # Recurrent net
        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')

        # Attention module
        self.attention = SequenceMultiContentAttention(
            state_names=self.transition.apply.states,
            attended_dims=[representation_dim, state_dim, state_dim],
            match_dim=state_dim, name="attention")

        # Readout module
        readout = ReadoutWithMultiEncoder(
            source_names=['states', 'feedback', 'src_selector_rep',
                          self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim, name='maxout_bias').apply,
                 Maxout(num_pieces=2, name='maxout').apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False, name='softmax0').apply,
                 Linear(input_dim=embedding_dim, name='softmax1').apply]),
            merged_dim=state_dim)

        # Sequence generator, that wraps everyhinga above
        self.sequence_generator = SequenceGeneratorWithMultiContext(
            num_contexts=3,  # attended, src_selector, trg_selector
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence',
                         'src_selector_rep', 'trg_selector_rep'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask,
             src_selector_rep, trg_selector_rep):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        sg_inputs = {'mask': target_sentence_mask,
                     'outputs': target_sentence,
                     'attended_0': representation,
                     'attended_1': src_selector_rep,
                     'attended_2': trg_selector_rep,
                     'attended_mask': source_sentence_mask}
        cost = self.sequence_generator.cost_matrix(**sg_inputs)

        return (cost * target_sentence_mask).sum() / target_sentence_mask.shape[1]

    @application
    def generate(self, source_sentence, representation):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            glimpses=self.attention.take_glimpses.outputs[0])


def main(config, tr_stream, dev_stream):

    # Create Theano variables
    src_selector = tensor.vector('src_selector', dtype=theano.config.floatX)
    trg_selector = tensor.vector('trg_selector', dtype=theano.config.floatX)
    source_sentences = [tensor.lmatrix('source_0'), tensor.lmatrix('source_1')]
    source_masks = [tensor.matrix('source_0_mask'), tensor.matrix('source_1_mask')]
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')

    # Construct model
    multi_encoder = MultiEncoder(config)
    decoder = Decoder(config['trg_vocab_size'], config['dec_embed'],
                      config['dec_nhids'], config['state_dim'],
                      src_selector, trg_selector, config['num_encs'])
    representation, src_mask, src_selector_rep, trg_selector_rep =\
        multi_encoder.apply(source_sentences, source_masks,
                            src_selector, trg_selector)

    cost = decoder.cost(
        representation, src_mask, target_sentence, target_sentence_mask,
        src_selector_rep, trg_selector_rep)

    # Initialize model
    multi_encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    multi_encoder.biases_init = Constant(0)
    multi_encoder.push_initialization_config()
    for i, _ in enumerate(source_sentences):
        multi_encoder.encoders[i].bidir.prototype.weights_init = Orthogonal()
    multi_encoder.initialize()
    decoder.weights_init = IsotropicGaussian(config['weight_scale'])
    decoder.biases_init = Constant(0)
    decoder.push_initialization_config()
    decoder.transition.weights_init = Orthogonal()
    decoder.initialize()

    cg = ComputationGraph(cost)

    # This block evaluates the encoder annotations
    import numpy
    f = theano.function(source_sentences + source_masks + [src_selector],
                        representation)
    s1 = numpy.random.randint(10, size=(10, 20))
    s2 = numpy.random.randint(10, size=(10, 30))
    m1 = numpy.random.rand(10, 20).astype('float32')
    m2 = numpy.random.rand(10, 30).astype('float32')
    rep_ = f(s1, s2, m1, m2, [1, 0])  # this should be of size (20,10,200)
    rep_ = f(s1, s2, m1, m2, [0, 1])  # this should be of size (30,10,200)

    # Set up training algorithm
    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=CompositeRule([StepClipping(config['step_clipping']),
                                 eval(config['step_rule'])()])
    )

    # Set up training model
    training_model = Model(cost)

    # Set extensions
    extensions = [
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        stream_fide_en.PrintMultiStream(after_batch=True),
        Dump(config['saveto'], every_n_batches=config['save_freq'])
    ]

    # Initialize main loop
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions
    )

    # Train!
    main_loop.run()
    print 'done!'

if __name__ == "__main__":
    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    tr_stream = streams[config['stream']].multi_enc_stream
    main(config, tr_stream, None)
