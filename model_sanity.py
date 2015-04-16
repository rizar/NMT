# This is the RNNsearch model
# Works with https://github.com/orhanf/blocks/tree/wmt15
# 0e23b0193f64dc3e56da18605d53d6f5b1352848
from collections import Counter
import argparse
import numpy
import os
import cPickle
import theano
from theano import tensor
from toolz import merge
from picklable_itertools.extras import equizip

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.dump import load_parameter_values
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import Checkpoint, LoadFromDump
from blocks.extensions.plot import Plot

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator
)

from stream_fi_en import masked_stream, state, dev_stream
from sampling import BleuValidator, Sampler


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


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


class Decoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim

        self.transition = GRUInitialState(
            attended_dim=state_dim, dim=state_dim,
            activation=Tanh(), name='decoder')
        self.attention = SequenceContentAttention(
            state_names=self.transition.apply.states,
            attended_dim=representation_dim,
            match_dim=state_dim, name="attention")

        readout = Readout(
            source_names=['states', 'feedback', self.attention.take_glimpses.outputs[0]],
            readout_dim=self.vocab_size,
            emitter=SoftmaxEmitter(initial_output=-1),
            feedback_brick=LookupFeedbackWMT15(vocab_size, embedding_dim),
            post_merge=InitializableFeedforwardSequence(
                [Bias(dim=state_dim).apply,
                 Maxout(num_pieces=2).apply,
                 Linear(input_dim=state_dim / 2, output_dim=embedding_dim,
                        use_bias=False).apply,
                 Linear(input_dim=embedding_dim).apply]),
            merged_dim=state_dim)

        self.sequence_generator = SequenceGenerator(
            readout=readout,
            transition=self.transition,
            attention=self.attention,
            fork=Fork([name for name in self.transition.apply.sequences
                       if name != 'mask'], prototype=Linear())
        )

        self.children = [self.sequence_generator]

    @application(inputs=['representation', 'source_sentence_mask',
                         'target_sentence_mask', 'target_sentence'],
                 outputs=['cost'])
    def cost(self, representation, source_sentence_mask,
             target_sentence, target_sentence_mask):

        source_sentence_mask = source_sentence_mask.T
        target_sentence = target_sentence.T
        target_sentence_mask = target_sentence_mask.T

        # Get the cost matrix
        cost = self.sequence_generator.cost_matrix(
                    **{'mask': target_sentence_mask,
                       'outputs': target_sentence,
                       'attended': representation,
                       'attended_mask': source_sentence_mask}
        )

        return (cost * target_sentence_mask).sum() / target_sentence_mask.shape[1]

    @application
    def generate(self, source_sentence, representation):
        return self.sequence_generator.generate(
            n_steps=2 * source_sentence.shape[1],
            batch_size=source_sentence.shape[0],
            attended=representation,
            attended_mask=tensor.ones(source_sentence.shape).T,
            glimpses=self.attention.take_glimpses.outputs[0])


if __name__ == "__main__":

    # Create Theano variables
    source_sentence = tensor.lmatrix('finnish')
    source_sentence_mask = tensor.matrix('finnish_mask')
    target_sentence = tensor.lmatrix('english')
    target_sentence_mask = tensor.matrix('english_mask')
    sampling_input = tensor.lmatrix('input')

    # Test values
    '''
    theano.config.compute_test_value = 'warn'
    source_sentence.tag.test_value = numpy.random.randint(10, size=(10, 10))
    target_sentence.tag.test_value = numpy.random.randint(10, size=(10, 10))
    source_sentence_mask.tag.test_value = \
        numpy.random.rand(10, 10).astype('float32')
    target_sentence_mask.tag.test_value = \
        numpy.random.rand(10, 10).astype('float32')
    sampling_input.tag.test_value = numpy.random.randint(10, size=(10, 10))
    '''

    # Construct model
    encoder = BidirectionalEncoder(state['src_vocab_size'], state['enc_embed'],
                                   state['enc_nhids'])
    decoder = Decoder(state['trg_vocab_size'], state['dec_embed'],
                      state['dec_nhids'], state['enc_nhids'] * 2)
    cost = decoder.cost(encoder.apply(source_sentence, source_sentence_mask),
                        source_sentence_mask, target_sentence, target_sentence_mask)

    # Initialize model
    encoder.weights_init = decoder.weights_init = IsotropicGaussian(state['weight_scale'])
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
        step_rule=CompositeRule([StepClipping(state['step_clipping']),
                                 eval(state['step_rule'])()])
    )

    # Set up beam search
    '''
    theano.config.compute_test_value = 'warn'
    sampling_input.tag.test_value = numpy.random.randint(10, size=(5, 7))
    '''
    sampling_encoder = BidirectionalEncoder(
        state['src_vocab_size'], state['enc_embed'], state['enc_nhids'])
    sampling_decoder = Decoder(state['trg_vocab_size'], state['dec_embed'],
                               state['dec_nhids'], state['enc_nhids'] * 2)
    sampling_encoder.weights_init = sampling_decoder.weights_init = Constant(0)
    sampling_encoder.biases_init = sampling_decoder.biases_init = Constant(0)
    sampling_representation = sampling_encoder.apply(
        sampling_input, tensor.ones(sampling_input.shape))
    generated = sampling_decoder.generate(
        sampling_input, sampling_representation)
    search_model = Model(generated)
    samples, = VariableFilter(
        bricks=[sampling_decoder.sequence_generator], name="outputs")(
            ComputationGraph(generated[1]))  # generated[1] is the next_outputs

    # Set up training model
    training_model = Model(cost)

    from blocks.select import Selector
    enc_param_dict = Selector(encoder).get_params()
    dec_param_dict = Selector(decoder).get_params()

    gh_model_name = '/data/lisatmp3/firatorh/nmt/wmt15/trainedModels/withoutLM/refGHOG_best_bleu_model.npz'

    tmp_file = numpy.load(gh_model_name)
    gh_model = dict(tmp_file)
    tmp_file.close()

    for key in enc_param_dict:
        print '{:15}: {}'.format(enc_param_dict[key].get_value().shape, key)
    for key in dec_param_dict:
        print '{:15}: {}'.format(dec_param_dict[key].get_value().shape, key)

    enc_param_dict['/bidirectionalencoder/embeddings.W'].set_value(gh_model['W_0_enc_approx_embdr'] + gh_model['b_0_enc_approx_embdr'])

    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/forward.state_to_state'].set_value(gh_model['W_enc_transition_0'])
    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/forward.state_to_update'].set_value(gh_model['G_enc_transition_0'])
    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/forward.state_to_reset'].set_value(gh_model['R_enc_transition_0'])

    enc_param_dict['/bidirectionalencoder/fwd_fork/fork_inputs.W'].set_value(gh_model['W_0_enc_input_embdr_0'])
    enc_param_dict['/bidirectionalencoder/fwd_fork/fork_inputs.b'].set_value(gh_model['b_0_enc_input_embdr_0'])
    enc_param_dict['/bidirectionalencoder/fwd_fork/fork_update_inputs.W'].set_value(gh_model['W_0_enc_update_embdr_0'])
    enc_param_dict['/bidirectionalencoder/fwd_fork/fork_reset_inputs.W'].set_value(gh_model['W_0_enc_reset_embdr_0'])

    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/backward.state_to_state'].set_value(gh_model['W_back_enc_transition_0'])
    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/backward.state_to_update'].set_value(gh_model['G_back_enc_transition_0'])
    enc_param_dict['/bidirectionalencoder/bidirectionalwmt15/backward.state_to_reset'].set_value(gh_model['R_back_enc_transition_0'])

    enc_param_dict['/bidirectionalencoder/back_fork/fork_inputs.W'].set_value(gh_model['W_0_back_enc_input_embdr_0'])
    enc_param_dict['/bidirectionalencoder/back_fork/fork_inputs.b'].set_value(gh_model['b_0_back_enc_input_embdr_0'])
    enc_param_dict['/bidirectionalencoder/back_fork/fork_update_inputs.W'].set_value(gh_model['W_0_back_enc_update_embdr_0'])
    enc_param_dict['/bidirectionalencoder/back_fork/fork_reset_inputs.W'].set_value(gh_model['W_0_back_enc_reset_embdr_0'])

    dec_param_dict['/decoder/sequencegenerator/readout/lookupfeedbackwmt15/lookuptable.W'].set_value(gh_model['W_0_dec_approx_embdr'] + gh_model['b_0_dec_approx_embdr'])

    dec_param_dict['/decoder/sequencegenerator/readout/initializablefeedforwardsequence/bias.b'].set_value(gh_model['b_0_dec_hid_readout_0'])
    decoder.children[0].children[0].children[3].children[2].params[0].set_value(gh_model['W1_dec_deep_softmax']) # Missing W1
    dec_param_dict['/decoder/sequencegenerator/readout/initializablefeedforwardsequence/linear.W'].set_value(gh_model['W2_dec_deep_softmax'])
    dec_param_dict['/decoder/sequencegenerator/readout/initializablefeedforwardsequence/linear.b'].set_value(gh_model['b_dec_deep_softmax'])

    dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_states.W'].set_value(gh_model['W_0_dec_hid_readout_0'])
    dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_feedback.W'].set_value(gh_model['W_0_dec_prev_readout_0'])
    dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_weighted_averages.W'].set_value(gh_model['W_0_dec_repr_readout'])

    dec_param_dict['/decoder/sequencegenerator/fork/fork_inputs.b'].set_value(gh_model['b_0_dec_input_embdr_0'])
    dec_param_dict['/decoder/sequencegenerator/fork/fork_inputs.W'].set_value(gh_model['W_0_dec_input_embdr_0'])
    dec_param_dict['/decoder/sequencegenerator/fork/fork_update_inputs.W'].set_value(gh_model['W_0_dec_update_embdr_0'])
    dec_param_dict['/decoder/sequencegenerator/fork/fork_reset_inputs.W'].set_value(gh_model['W_0_dec_reset_embdr_0'])

    dec_param_dict['/decoder/sequencegenerator/att_trans/distribute/fork_inputs.W'].set_value(gh_model['W_0_dec_dec_inputter_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/distribute/fork_update_inputs.W'].set_value(gh_model['W_0_dec_dec_updater_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/distribute/fork_reset_inputs.W'].set_value(gh_model['W_0_dec_dec_reseter_0'])

    dec_param_dict['/decoder/sequencegenerator/att_trans/decoder.state_to_state'].set_value(gh_model['W_dec_transition_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/decoder.state_to_update'].set_value(gh_model['G_dec_transition_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/decoder.state_to_reset'].set_value(gh_model['R_dec_transition_0'])

    dec_param_dict['/decoder/sequencegenerator/att_trans/attention/state_trans/transform_states.W'].set_value(gh_model['B_dec_transition_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/attention/preprocess.W'].set_value(gh_model['A_dec_transition_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/attention/energy_comp/mlp/linear_0.W'].set_value(gh_model['D_dec_transition_0'])

    dec_param_dict['/decoder/sequencegenerator/att_trans/decoder/state_initializer/linear_0.W'].set_value(gh_model['W_0_dec_initializer_0'])
    dec_param_dict['/decoder/sequencegenerator/att_trans/decoder/state_initializer/linear_0.b'].set_value(gh_model['b_0_dec_initializer_0'])

    # Initialize main loop
    main_loop = MainLoop(
        model=training_model,
        algorithm=algorithm,
        data_stream=masked_stream,
        extensions=[
            #LoadFromDump(state['prefix'] + 'model.pkl'),
            Sampler(model=search_model, state=state, data_stream=masked_stream,
                    every_n_batches=state['sampling_freq']),
            BleuValidator(sampling_input, samples=samples, state=state,
                          model=search_model, data_stream=dev_stream,
                          before_batch=True), #every_n_batches=state['bleu_val_freq']),
            TrainingDataMonitoring([cost], after_batch=True),
            #Plot('En-Fr', channels=[['decoder_cost_cost']],
            #     after_batch=True),
            Printing(after_batch=True),
            Checkpoint(state['prefix'] + 'model.pkl',
                       every_n_batches=state['save_freq'])
        ]
    )

    # Reload model
    file_to_load = state['prefix'] + 'model.pkl'
    if state['reload'] and os.path.isfile(file_to_load):
        main_loop = cPickle.load(open(file_to_load))

    # Train!
    main_loop.run()