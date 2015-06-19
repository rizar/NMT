# This is the RNNsearch model
# Works with https://github.com/orhanf/blocks/tree/wmt15
# 0e23b0193f64dc3e56da18605d53d6f5b1352848
import argparse
import importlib
import logging
import pprint
import theano
from collections import Counter
from theano import tensor
from toolz import merge

from blocks.algorithms import (AdaDelta, CompositeRule, Adam, RemoveNotFinite,
                               StepClipping)
from blocks.filter import VariableFilter
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.plot import Plot

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, Identity)
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.select import Selector
from blocks.bricks.sequence_generators import Readout, SoftmaxEmitter

import config
import multiCG_stream

from multiCG_algorithm import (GradientDescentWithMultiCG,
                               MainLoopWithMultiCG)
from multiCG_attention import SequenceMultiContentAttention
from multiCG_extensions import (TrainingDataMonitoringWithMultiCG,
                                DumpWithMultiCG,
                                LoadFromDumpMultiCG)
from multiCG_recurrent import BidirectionalWMT15, GRUwithContext
from multiCG_sequence_generator import (
    LookupFeedbackWMT15, SequenceGeneratorWithMultiContext)

from sampling import Sampler, BleuValidator

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_config_wmt15_fide_en_TEST",
                    help="Prototype config to use for config")
args = parser.parse_args()

# Make config global, nasty workaround since parameterizing stream
# will cause erroneous picklable behaviour, find a better solution
config = getattr(config, args.proto)()


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


class MultiEncoder(Initializable):

    def __init__(self, num_encs, num_decs, src_vocab_sizes, enc_embed_sizes,
                 enc_nhids, representation_dim, **kwargs):
        super(MultiEncoder, self).__init__(**kwargs)

        self.num_encs = num_encs
        self.encoders = []
        for i in xrange(self.num_encs):
            self.encoders.append(
                BidirectionalEncoder(
                    src_vocab_sizes[i],
                    enc_embed_sizes[i],
                    enc_nhids[i],
                    enc_id=i))

        # this is the embedding from h to z
        self.annotation_embedders = [Linear(input_dim=(2 * enc_nhids[i]),
                                            output_dim=representation_dim,
                                            name='annotation_embedder_%d' % i,
                                            use_bias=False)
                                     for i in xrange(self.num_encs)]

        self.src_selector_embedder = Identity(name='src_selector_embedder')
        self.trg_selector_embedder = Identity(name='trg_selector_embedder')

        self.children = self.encoders + self.annotation_embedders + \
            [self.src_selector_embedder, self.trg_selector_embedder]

    @application
    def apply(self, source_sentence, source_mask,
              src_selector, trg_selector, enc_idx):

        # Projected Annotations
        rep = self.annotation_embedders[enc_idx].apply(
            self.encoders[enc_idx].apply(source_sentence, source_mask))

        # Source selector annotations, expand it to have batch size
        # dimensions for further ease in recurrence
        src_selector_rep = self.src_selector_embedder.apply(
            theano.tensor.repeat(
                src_selector[None, :], rep.shape[1], axis=0)
        )
        # Target selector annotations, expand it similarly
        trg_selector_rep = self.trg_selector_embedder.apply(
            theano.tensor.repeat(
                trg_selector[None, :], rep.shape[1], axis=0)
        )
        return rep, src_selector_rep, trg_selector_rep


class BidirectionalEncoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim, enc_id, **kwargs):
        super(BidirectionalEncoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.lookup = LookupTable(name='embeddings')
        self.bidir = BidirectionalWMT15(GatedRecurrent(activation=Tanh(),
                                                       dim=state_dim))
        self.enc_id = enc_id
        self.name = 'bidirectionalencoder_%d' % enc_id

        self.fwd_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='fwd_fork')
        self.back_fork = Fork(
            [name for name in self.bidir.prototype.apply.sequences
             if name != 'mask'], prototype=Linear(), name='back_fork')

        self.children = [self.lookup, self.bidir,
                         self.fwd_fork, self.back_fork]

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


class Decoder(Initializable):
    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, num_encs, num_decs, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.num_encs = num_encs
        self.num_decs = num_decs

        # Recurrent net
        self.transition = GRUwithContext(
            attended_dim=state_dim, dim=state_dim, context_dim=num_encs,
            activation=Tanh(), name='decoder')

        # Attention module
        self.attention = SequenceMultiContentAttention(
            state_names=self.transition.apply.states,
            attended_dims=[representation_dim, num_encs, num_decs],
            match_dim=state_dim, name="attention")

        # Readout module
        readout = Readout(
            source_names=['states', 'feedback', 'attended_1',
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

        return (cost * target_sentence_mask).sum() / \
            target_sentence_mask.shape[1]

    @application
    def generate(self, source_sentence, representation,
                 src_selector_rep, trg_selector_rep):

        n_steps = source_sentence.shape[1]
        batch_size = source_sentence.shape[0]
        attended_mask = tensor.ones(source_sentence.shape).T

        return self.sequence_generator.generate(
            n_steps=2 * n_steps,
            batch_size=batch_size,
            attended_0=representation,
            attended_1=src_selector_rep,
            attended_2=trg_selector_rep,
            attended_mask=attended_mask)

    @application
    def get_decoder_transition(
            self, representation, source_sentence_mask,
            target_sentence_mask, target_sentence, src_selector_rep,
            trg_selector_rep):

            sg_inps = {'mask': target_sentence_mask.T,
                       'outputs': target_sentence.T,
                       'attended_0': representation,
                       'attended_1': src_selector_rep,
                       'attended_2': trg_selector_rep,
                       'attended_mask': source_sentence_mask.T}
            return self.sequence_generator.get_transition_func(**sg_inps)


def main(config, tr_stream, dev_streams):

    # Create Theano variables
    # Training
    src_selector = tensor.vector('src_selector', dtype=theano.config.floatX)
    trg_selector = tensor.vector('trg_selector', dtype=theano.config.floatX)
    source_sentence = tensor.lmatrix('source')
    source_mask = tensor.matrix('source_mask')
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')

    # Sampling
    sampling_input = tensor.lmatrix('input')
    sampling_mask = tensor.ones(sampling_input.shape)
    sampling_src_sel = tensor.vector('sampling_src_sel',
                                     dtype=theano.config.floatX)
    sampling_trg_sel = tensor.vector('sampling_trg_sel',
                                     dtype=theano.config.floatX)

    # Construct model
    multi_encoder = MultiEncoder(
        num_encs=config['num_encs'],
        num_decs=config['num_decs'],
        representation_dim=config['representation_dim'],
        src_vocab_sizes=[config['src_vocab_size_%d' % ii]
                         for ii in xrange(config['num_encs'])],
        enc_embed_sizes=[config['enc_embed_%d' % ii]
                         for ii in xrange(config['num_encs'])],
        enc_nhids=[config['enc_nhids_%d' % ii]
                   for ii in xrange(config['num_encs'])])
    decoder = Decoder(
        vocab_size=config['trg_vocab_size'],
        embedding_dim=config['dec_embed'],
        state_dim=config['dec_nhids'],
        representation_dim=config['representation_dim'],
        num_encs=config['num_encs'],
        num_decs=config['num_decs'])

    # Get costs from each encoder sources
    costs = []
    for i in xrange(config['num_encs']):
        representation, src_selector_rep, trg_selector_rep =\
            multi_encoder.apply(source_sentence, source_mask,
                                src_selector, trg_selector, i)
        costs.append(
            decoder.cost(
                representation, source_mask,
                target_sentence, target_sentence_mask,
                src_selector_rep, trg_selector_rep))
        costs[i].name += "_{}".format(i)

    # Initialize model
    multi_encoder.weights_init = IsotropicGaussian(config['weight_scale'])
    multi_encoder.biases_init = Constant(0)
    multi_encoder.push_initialization_config()
    for i in xrange(config['num_encs']):
        multi_encoder.encoders[i].bidir.prototype.weights_init = Orthogonal()
    multi_encoder.initialize()
    decoder.weights_init = IsotropicGaussian(config['weight_scale'])
    decoder.biases_init = Constant(0)
    decoder.push_initialization_config()
    decoder.transition.weights_init = Orthogonal()
    decoder.initialize()

    # Get computation graphs
    cgs = []
    for i in xrange(config['num_encs']):
        cgs.append(ComputationGraph(costs[i]))

        # Print shapes
        shapes = [param.get_value().shape for param in cgs[i].parameters]
        logger.info("Parameter shapes for computation graph[{}]".format(i))
        for shape, count in Counter(shapes).most_common():
            logger.info('    {:15}: {}'.format(shape, count))
        logger.info(
            "Total number of parameters for computation graph[{}]: {}"
            .format(i, len(shapes)))

        logger.info("Parameter names for computation graph[{}]: ".format(i))
        enc_dec_param_dict = merge(
            Selector(multi_encoder.encoders[i]).get_params(),
            Selector(multi_encoder.annotation_embedders[i]).get_params(),
            Selector(multi_encoder.src_selector_embedder).get_params(),
            Selector(multi_encoder.trg_selector_embedder).get_params(),
            Selector(decoder).get_params())
        for name, value in enc_dec_param_dict.iteritems():
            logger.info('    {:15}: {}'.format(value.get_value().shape, name))
        logger.info("Total number of parameters for computation graph[{}]: {}"
                    .format(i, len(enc_dec_param_dict)))

    # Print parameter names
    enc_dec_param_dict = merge(Selector(multi_encoder).get_params(),
                               Selector(decoder).get_params())
    logger.info("Parameter names: ")
    for name, value in enc_dec_param_dict.iteritems():
        logger.info('    {:15}: {}'.format(value.get_value().shape, name))
    logger.info(
        "Total number of parameters: {}".format(len(enc_dec_param_dict)))

    # Exclude additional parameters from training if any
    # Note that, additional excludes are excluded from all computatinal graphs
    excluded_params = [list() for _ in xrange(len(cgs))]
    if 'additional_excludes' in config:

        # Get parameters to exclude first
        pex = [enc_dec_param_dict[p] for p in config['additional_excludes']
               if p in enc_dec_param_dict]

        # Put parameter into exclude list
        for i in xrange(config['num_encs']):
            for p in pex:
                if p in cgs[i].parameters:
                    excluded_params[i].append(p)

    # Exclude encoder parameters from training
    training_params = []
    if 'exclude_encs' in config:
        assert config['num_encs'] == len(config['exclude_encs']), \
            "Erroneous config::[num_encs] should match [exclude_encs]"
        for i in xrange(config['num_encs']):
            if config['exclude_encs'][i]:
                p_enc = Selector(multi_encoder.encoders[i]).get_params()
                training_params.append(
                    [p for p in cgs[i].parameters
                        if (not any([pp == p for pp in p_enc.values()])) and
                           (p not in excluded_params[i])])
            else:
                training_params.append([p for p in cgs[i].parameters
                                        if p not in excluded_params[i]])

    # Print which parameters are excluded
    for i in xrange(config['num_encs']):
        excluded_all = list(set(cgs[i].parameters) - set(training_params[i]))
        for p in excluded_all:
            logger.info(
                'Excluding from training of CG[{}]: [{}]'
                .format(i, [key for key, val in enc_dec_param_dict.iteritems()
                            if val == p][0]))
        logger.info(
            'Total number of excluded parameters for CG[{}]: [{}]'
            .format(i, len(excluded_all)))

    # Set up training algorithm
    algorithm = GradientDescentWithMultiCG(
        costs=costs, params=training_params, drop_input=config['drop_input'],
        step_rule=CompositeRule(
            [StepClipping(threshold=config['step_clipping']),
             RemoveNotFinite(0.9),
             eval(config['step_rule'])(
                 learning_rate=config['learning_rate'])]))

    # Set up training model
    training_models = []
    for i in xrange(config['num_encs']):
        training_models.append(Model(costs[i]))

    # Set observables for monitoring
    observables = costs
    observables = [[x] for x in observables]

    # Set extensions
    extensions = [
        FinishAfter(after_n_batches=config['finish_after']),
        TrainingDataMonitoringWithMultiCG(observables, after_batch=True),
        Printing(after_batch=True),
        multiCG_stream.PrintMultiStream(after_batch=True),
        DumpWithMultiCG(saveto=config['saveto'],
                        save_accumulators=config['save_accumulators'],
                        every_n_batches=config['save_freq'])]

    # Set up beam search and sampling computation graphs
    for i in xrange(config['num_encs']):

        # Compute annotations from one of the encoders
        sampling_rep, src_selector_rep, trg_selector_rep =\
            multi_encoder.apply(sampling_input, sampling_mask,
                                sampling_src_sel, sampling_trg_sel, i)

        # Get sampling computation graph
        generated = decoder.generate(sampling_input, sampling_rep,
                                     src_selector_rep, trg_selector_rep)

        # Filter the output variable that corresponds to the sample
        # generated[1] is the next_outputs
        samples, = VariableFilter(
            bricks=[decoder.sequence_generator], name="outputs")(
                ComputationGraph(generated[1]))

        # Create a model for the computation graph
        sampling_model = Model(generated)

        # Add sampling for multi encoder
        extensions.append(Sampler(
            sampling_model, tr_stream, num_samples=config['hook_samples'],
            src_eos_idx=config['src_eos_idx_%d' % i],
            trg_eos_idx=config['trg_eos_idx'], enc_id=i,
            every_n_batches=config['sampling_freq']))

        # Add bleu validator for multi encoder, except for the identical
        # mapping languages such as english-to-english computation graph
        if config['src_data_%d' % i] != config['trg_data_%d' % i]:
            extensions.append(BleuValidator(
                samples, sampling_model, dev_streams[i],
                src_selector=sampling_src_sel, trg_selector=sampling_trg_sel,
                src_vocab_size=config['src_vocab_size_%d' % i],
                bleu_script=config['bleu_script'],
                val_set_out=config['val_set_out_%d' % i],
                val_set_grndtruth=config['val_set_grndtruth_%d' % i],
                beam_size=config['beam_size'],
                val_burn_in=config['val_burn_in'],
                enc_id=i, saveto=config['saveto'],
                track_n_models=config['track_n_models'],
                src_eos_idx=config['src_eos_idx_%d' % i],
                trg_eos_idx=config['trg_eos_idx'],
                every_n_batches=config['bleu_val_freq']))

    # Reload model if necessary
    if config['reload']:
        extensions.append(
            LoadFromDumpMultiCG(saveto=config['saveto'],
                                load_accumulators=config['load_accumulators']))

    # This is for bokeh server, highly not recommended
    if config['plot']:
        extensions.append(
            Plot(config['stream'],
                 channels=[['decoder_cost_cost_%d' % i]
                           for i in xrange(config['num_encs'])],
                 server_url="http://127.0.0.1:{}".format(config['bokeh_port']),
                 after_batch=True))

    # Initialize main loop
    main_loop = MainLoopWithMultiCG(
        models=training_models,
        algorithm=algorithm,
        data_stream=tr_stream,
        extensions=extensions)

    # Train!
    main_loop.run()
    print 'done!'

if __name__ == "__main__":
    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    stream = importlib.import_module(config['stream'])
    main(config, stream.multi_enc_stream, stream.dev_streams)
