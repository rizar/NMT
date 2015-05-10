#
# WARNING: SLOW STREAM
# Should probably use caching and multiprocessing like in the tutorial
# The files are those from WMT15, the vocab files are simply the 30,000 most
# common words of the raw data
#

import cPickle
import numpy
import six
import theano

from blocks.extensions import SimpleExtension

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Transformer, Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

# Everthing here should be wrapped and parameterized by config
# this import is to workaround for pickling errors when wrapped
from model_multi_enc import config

num_encs = config['num_encs']


class PrintMultiStream(SimpleExtension):
    """Prints number of batches seen for each data stream"""
    def __init__(self, **kwargs):
        super(PrintMultiStream, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        counters = self.main_loop.data_stream.training_counter
        sid = self.main_loop.data_stream.curr_id
        msg = ['source_{}:[{}]'.format(i, c) for i, c in enumerate(counters)]
        print("Multi-stream status:")
        print "\t", "Using stream: source_{}".format(sid)
        print "\t", " ".join(msg)


class MultiEncStream(Transformer, six.Iterator):

    def __init__(self, streams, schedule, batch_sizes):

        self.streams = streams
        self.schedule = numpy.asarray(schedule)
        self.counters = numpy.zeros_like(self.schedule)
        self.curr_id = 0  # this is the config of transformer
        self.curr_epoch_iterator = None
        self.num_encs = len(streams)
        self.training_counter = numpy.zeros_like(self.counters)
        self.batch_sizes = batch_sizes

        # Get all epoch iterators
        self.epoch_iterators = [st.get_epoch_iterator(as_dict=True)
                                for st in self.streams]

        # Initialize epoch iterator id to zero
        self.curr_epoch_iterator = self.epoch_iterators[0]

    def get_epoch_iterator(self, **kwargs):
        return self

    def __next__(self):
        batch = next(self.curr_epoch_iterator)
        batch['src_selector'] = self.curr_id
        self._add_selectors(batch)
        self._add_missing_sources(batch)
        self._update_counters()
        return batch

    def _add_selectors(self, batch):
        """Set src and target selector vectors"""
        batch['src_selector'] = numpy.zeros(
            (self.num_encs,)).astype(theano.config.floatX)
        batch['src_selector'][self.curr_id] = 1.
        batch['trg_selector'] = numpy.tile(
            1., (1,)).astype(theano.config.floatX)

    def _add_missing_sources(self, batch):

        # Find missing source language
        missing_idx = [k for k in xrange(self.num_encs)
                       if 'source_%d' % k not in batch.keys()]

        # Add sequence of -1  and mask of zeros
        for idx in missing_idx:
            ref_seq = batch['source_%d' % self.curr_id]
            ref_msk = batch['source_%d_mask' % self.curr_id]
            batch['source_%d' % idx] = numpy.zeros_like(ref_seq) * -1
            batch['source_%d_mask' % idx] = numpy.zeros_like(ref_msk) * 0.

    def _update_counters(self):
        # Increment counter and check schedule
        self.training_counter[self.curr_id] += 1
        self.counters[self.curr_id] += 1
        vict_idx = numpy.where(self.counters // self.schedule)[0]

        # Change stream
        if len(vict_idx):
            self.counters[self.curr_id] = 0
            self.curr_id = (vict_idx[0] + 1) % len(self.streams)
            self.curr_epoch_iterator = self.epoch_iterators[self.curr_id]


# If you wrap following functions, main_loop cannot be pickled ****************
def _length(sentence_pair):
    '''Assumes target is the last element in the tuple'''
    return len(sentence_pair[-1])


def _oov_to_unk(sentence_pair, src_vocab_size=30000,
                trg_vocab_size=30000, unk_id=1):
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def _too_long(sentence_pair, params):
    # TODO: harmonize argumenrs with oov_to_unk
    seq_len = 50
    if 'seq_len' in params:
        seq_len = params['seq_len']
    return all([len(sentence) < seq_len
                for sentence in sentence_pair])


def _oov_to_unk_multi(sentence_pair, src_vocab_sizes=None,
                      trg_vocab_size=30000, unk_id=1):
    return tuple([[x if x < src_vocab_sizes[i] else unk_id for x in sentence_pair[i]]
                  for i in xrange(len(src_vocab_sizes))] +
                 [[x if x < trg_vocab_size else unk_id for x in sentence_pair[1]]])
# *****************************************************************************

# Prepare source vocabs and files, there are 2 vocabs and 2 data files
src_vocabs = [config['src_vocab_%d' % x] for x in xrange(num_encs)]
src_files = [config['src_data_%d' % x] for x in xrange(num_encs)]

# Prepare target vocabs and files, there are  2 vocabs and 3 data files
trg_vocab = config['trg_vocab']
trg_files = [config['trg_data_%d' % x] for x in xrange(num_encs)]

# Create individual source streams
src_datasets = [TextFile([ff], cPickle.load(open(vv)), None)
                for ff, vv in zip(src_files, src_vocabs)]

# Create individial target streams
trg_datasets = [TextFile([ff], cPickle.load(open(trg_vocab)), None)
                for ff in trg_files]

# Build the preprocessing pipeline for individual streams
ind_streams = []
for i in xrange(num_encs):
    stream = Merge([src_datasets[i].get_example_stream(),
                    trg_datasets[i].get_example_stream()],
                   ('source_%d' % i, 'target'))
    stream = Filter(stream, predicate=_too_long,
                    predicate_args={'seq_len':config['seq_len']})
    stream = Mapping(stream, _oov_to_unk,
                     src_vocab_size=config['src_vocab_size_%d' % i],
                     trg_vocab_size=config['trg_vocab_size'],
                     unk_id=config['unk_id'])
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       config['batch_size_enc_%d' % i]*config['sort_k_batches']))

    stream = Mapping(stream, SortMapping(_length))
    stream = Unpack(stream)
    stream = Batch(stream, iteration_scheme=ConstantScheme(
        config['batch_size_enc_%d' % i]))
    masked_stream = Padding(stream)
    ind_streams.append(masked_stream)

multi_enc_stream = MultiEncStream(ind_streams, schedule=config['schedule'],
                                  batch_sizes=[config['batch_size_enc_%d' % i]
                                              for i in xrange(num_encs)]  )
