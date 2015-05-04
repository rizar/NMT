# This is the RNNsearch model
# Works with https://github.com/orhanf/blocks/tree/wmt15
# 0e23b0193f64dc3e56da18605d53d6f5b1352848
from collections import Counter
import argparse
import logging
import pprint
import theano
from theano import tensor
from toolz import merge
from picklable_itertools.extras import equizip

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.dump import MainLoopDumpManager
from blocks.filter import VariableFilter
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.graph import ComputationGraph, apply_noise, apply_dropout
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import LoadFromDump, Dump
from blocks.extensions.plot import Plot

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence,
                           Bias, Initializable, MLP)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent, Bidirectional
from blocks.select import Selector
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter,
    SequenceGenerator
)
from blocks.select import Selector

import states
import stream_fide_en

from sampling import BleuValidator, Sampler

logger = logging.getLogger(__name__)

# Get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proto",  default="get_states_wmt15_fide_en_TEST",
                    help="Prototype state to use for state")
args = parser.parse_args()

# Make state global, nasty workaround since parameterizing stream
# will cause erroneous picklable behaviour, find a better solution
state = getattr(states, args.proto)()

# dictionary mapping stream name to stream getters
streams = {'fide-en': stream_fide_en}


class MultiEncoder(Initializable):

    def __init__(self, state, source_sentences, source_masks, **kwargs):
        super(MultiEncoder, self).__init__(**kwargs)

        self.num_encs = len(source_sentences)
        self.encoders = []
        for i in xrange(self.num_encs):
            self.encoders.append(
                BidirectionalEncoder(
                    state['src_vocab_size_%d' % i],
                    state['enc_embed_%d' % i],
                    state['enc_nhids_%d' % i]))

        self.children = self.encoders

    @application
    def apply(self, source_sentences, source_masks):
        representations = []
        for i, (sentence, mask) in enumerate(
                zip(source_sentences, source_masks)):
            # TODO: add the condition to check if batch has all -1 meaning the
            # source batch does not exist, and set representation to zero
            representations.append(self.encoders[i].apply(sentence, mask))
        return representations


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


def main(state, tr_stream, dev_stream):

    # Create Theano variables
    source_sentences = [tensor.lmatrix('source_0'), tensor.lmatrix('source_1')]
    source_masks = [tensor.matrix('source_0_mask'), tensor.matrix('source_1_mask')]
    target_sentence = tensor.lmatrix('target')
    target_sentence_mask = tensor.matrix('target_mask')

    # Construct model
    multi_encoder = MultiEncoder(state, source_sentences, source_masks)
    representations = multi_encoder.apply(source_sentences, source_masks)

    # Initialize model
    multi_encoder.weights_init = IsotropicGaussian(state['weight_scale'])
    multi_encoder.biases_init = Constant(0)
    multi_encoder.push_initialization_config()
    for i, _ in enumerate(source_sentences):
        multi_encoder.encoders[i].bidir.prototype.weights_init = Orthogonal()
    multi_encoder.initialize()

    # This block evaluates the encoder annotations
    f = theano.function(source_sentences + source_masks, representations)
    import numpy
    s1 = numpy.random.randint(10, size=(10, 20))
    s2 = numpy.random.randint(10, size=(10, 30))
    m1 = numpy.random.rand(10, 20).astype('float32')
    m2 = numpy.random.rand(10, 30).astype('float32')
    rep_ = f(s1, s2, m1, m2)

    # Create a dummy cost
    cost = theano.tensor.concatenate(representations, axis=0).sum()

    cg = ComputationGraph(cost)

    # Set up training algorithm
    algorithm = GradientDescent(
        cost=cost, params=cg.parameters,
        step_rule=CompositeRule([StepClipping(state['step_clipping']),
                                 eval(state['step_rule'])()])
    )

    # Set up training model
    training_model = Model(cost)

    # Set extensions
    extensions = [
        TrainingDataMonitoring([cost], after_batch=True),
        Printing(after_batch=True),
        Dump(state['saveto'], every_n_batches=state['save_freq'])
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
    logger.info("Model options:\n{}".format(pprint.pformat(state)))
    tr_stream = streams[state['stream']].multi_enc_stream
    main(state, tr_stream, None)

