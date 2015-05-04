#
# WARNING: SLOW STREAM
# Should probably use caching and multiprocessing like in the tutorial
# The files are those from WMT15, the vocab files are simply the 30,000 most
# common words of the raw data
#

import cPickle
import numpy
import six

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Transformer, Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

# Everthing here should be wrapped and parameterized by state
# this import is to workaround for pickling errors when wrapped
from model_multi_enc import state

num_encs = state['num_encs']


class MultiEncStream(Transformer, six.Iterator):

    def __init__(self, streams, schedule):
        '''Assumes joint stream is the last one'''

        self.streams = streams
        self.schedule = numpy.asarray(schedule)
        self.counter = numpy.zeros_like(self.schedule)
        self.curr_id = 0  # this is the state of transformer
        self.curr_epoch_iterator = None
        self.num_encs = len(streams) - 1

        # Get all epoch iterators
        self.epoch_iterators = [self.streams[i]
                                .get_epoch_iterator(as_dict=True)
                                for i in xrange(self.num_encs)]

        # Initialize epoch iterator id to zero
        self.curr_epoch_iterator = self.epoch_iterators[0]

    def get_epoch_iterator(self, **kwargs):
        return self

    def __next__(self):
        batch = next(self.curr_epoch_iterator)
        if self.curr_id >= 0:
            self._add_missing_source(batch)
        self._update_counter()

    def _add_missing_source(self, batch):

        # Find missing source language
        missing_idx = [k for k in xrange(self.num_encs)
                       if 'source_%d' % k not in batch.keys()]

        # Add sequence and mask of minus ones
        for idx in missing_idx:
            batch['source_%d' % k] = numpy.tile(
                    -1., batch['source_%d' % self.curr_id].shape)
            batch['source_mask_%d' % k] = numpy.tile(
                    -1., batch['source_%d' % self.curr_id].shape)

    def _update_counter(self):

        # Increment counter and check schedule
        self.counter[self.curr_id] += 1
        vict_idx = numpy.where(self.counter // self.schedule)[0]

        # Change stream
        if not vict_idx:
            self.curr_id = (vict_idx + 1) % len(self.streams)
            self.curr_epoch_iterator = self.epoch_iterators[self.curr_id]


# If you wrap following functions, main_loop cannot be pickled ****************
def _length(sentence_pair):
    '''Assumes target is the last element in the tuple'''
    return len(sentence_pair[-1])


def _oov_to_unk(sentence_pair, src_vocab_size=30000,
                trg_vocab_size=30000, unk_id=1):
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def _too_long(sentence_pair, seq_len=50):
    return all([len(sentence) < seq_len
                for sentence in sentence_pair])


def _oov_to_unk_multi(sentence_pair, src_vocab_sizes=None,
                      trg_vocab_size=30000, unk_id=1):
    return tuple([[x if x < src_vocab_sizes[i] else unk_id for x in sentence_pair[i]]
                  for i in xrange(len(src_vocab_sizes))] +
                 [[x if x < trg_vocab_size else unk_id for x in sentence_pair[1]]])
# *****************************************************************************

# Prepare source vocabs and files, there are 2 vocabs and 3 data files
src_vocabs = [state['src_vocab_%d' % x] for x in xrange(num_encs)]
src_files = [state['src_data_%d' % x] for x in xrange(num_encs)]

# Prepare target vocabs and files, there are  2 vocabs and 3 data files
trg_vocab = state['trg_vocab']
trg_files = [state['trg_data_%d' % x] for x in xrange(num_encs)]

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
                    predicate_args={'seq_len':state['seq_len']})
    stream = Mapping(stream, _oov_to_unk,
                     src_vocab_size=state['src_vocab_size_%d' % i],
                     trg_vocab_size=state['trg_vocab_size'],
                     unk_id=state['unk_id'])
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       state['batch_size_enc_%d' % i]*state['sort_k_batches']))

    stream = Mapping(stream, SortMapping(_length))
    stream = Unpack(stream)
    stream = Batch(stream, iteration_scheme=ConstantScheme(
        state['batch_size_enc_%d' % i]))
    masked_stream = Padding(stream)
    ind_streams.append(masked_stream)


# Build the joint stream !
joint_src_files = [state['src_data_joint_%d' % i]
                   for i in xrange(num_encs)]
joint_trg_file = state['trg_data_joint']

joint_src_datasets = [TextFile([ff], cPickle.load(open(vv)), None)
                      for ff, vv in zip(joint_src_files, src_vocabs)]

joint_trg_dataset = TextFile([joint_trg_file],
                             cPickle.load(open(trg_vocab)), None)

# Build the preprocessing pipeline for individaul streams
stream = Merge([st.get_example_stream()
                for st in joint_src_datasets + [joint_trg_dataset]],
               tuple(['source_%d' % ii for ii in xrange(num_encs)] +
                     ['target']))

stream = Filter(stream, predicate=_too_long,
                predicate_args={'seq_len': state['seq_len']})
stream = Mapping(stream, _oov_to_unk_multi,
                 src_vocab_sizes=[state['src_vocab_size_%d' % i]
                                  for i in xrange(num_encs)],
                 trg_vocab_size=state['trg_vocab_size'],
                 unk_id=state['unk_id'])
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   state['batch_size_joint']*state['sort_k_batches']))

stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(
               state['batch_size_joint']))
joint_stream = Padding(stream)

multi_enc_stream = MultiEncStream(ind_streams + [joint_stream],
                                  schedule=[1, 2, 3])
