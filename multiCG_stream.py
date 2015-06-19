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
        epochs = self.main_loop.data_stream.epoch_counter
        sid = self.main_loop.data_stream.curr_id
        src_size = args[0]['source'].shape
        trg_size = args[0]['target'].shape
        msg = ['Source_{}:iter[{}]-epoch[{}]'.format(i, c, e)
               for i, (c, e) in enumerate(zip(counters, epochs))]
        print("Multi-stream status:")
        print "\t", "Using stream: source_{}".format(sid)
        print "\t", "Source shape: {}".format(src_size)
        print "\t", "Target shape: {}".format(trg_size)
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
        self.epoch_counter = numpy.zeros_like(self.counters)
        self.batch_sizes = batch_sizes

        # Get all epoch iterators
        self.epoch_iterators = [st.get_epoch_iterator(as_dict=True)
                                for st in self.streams]

        # Initialize epoch iterator id to zero
        self.curr_epoch_iterator = self.epoch_iterators[0]

    def get_epoch_iterator(self, **kwargs):
        return self

    def __next__(self):
        batch = self._get_batch_with_reset(
            self.epoch_iterators[self.curr_id])
        self._add_selectors(batch, self.curr_id)
        self._update_counters()
        return batch

    def _add_selectors(self, batch, src_id):
        """Set src and target selector vectors"""
        batch['src_selector'] = numpy.zeros(
            (self.num_encs,)).astype(theano.config.floatX)
        batch['src_selector'][src_id] = 1.
        batch['trg_selector'] = numpy.tile(
            1., (1,)).astype(theano.config.floatX)

    def _add_missing_sources(self, batch, src_id):

        # Find missing source language
        missing_idx = [k for k in xrange(self.num_encs)
                       if 'source_%d' % k not in batch.keys()]

        # Add sequence of -1  and mask of zeros
        for idx in missing_idx:
            ref_seq = batch['source_%d' % src_id]
            ref_msk = batch['source_%d_mask' % src_id]
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

    def get_batches_from_all_streams(self):
        batches = []
        for i in xrange(self.num_encs):
            batch = self._get_batch_with_reset(self.epoch_iterators[i])
            self._add_selectors(batch, i)
            #self._add_missing_sources(batch, i)
            batches.append(batch)
        return batches

    def get_batch_with_stream_id(self, stream_id):
        batch = self._get_batch_with_reset(self.epoch_iterators[stream_id])
        self._add_selectors(batch, stream_id)
        return batch

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_batch_with_reset(self, epoch_iterator):
        while True:
            try:
                batch = next(epoch_iterator)
                return batch
            # TODO: This may not be the only source of exception
            except:
                sources = self._get_attr_rec(
                        epoch_iterator, 'data_stream').data_streams
                # Reset streams
                for st in sources:
                    st.reset()
                # Increment epoch counter
                self._update_epoch_counter(epoch_iterator)

    def _update_epoch_counter(self, epoch_iterator):
        idx = [i for i, t in enumerate(self.epoch_iterators)
               if t == epoch_iterator][0]
        self.epoch_counter[idx] += 1


def _length(sentence_pair):
    '''Assumes target is the last element in the tuple'''
    return len(sentence_pair[-1])


class _remapWordIdx(object):
    def __init__(self, mappings):
        self.mappings = mappings

    def __call__(self, sentence_pair):
        for mapping in self.mappings:
            sentence_pair[mapping[0]][numpy.where(
                sentence_pair[mapping[0]] == mapping[1])] = mapping[2]
        return sentence_pair


class _oov_to_unk(object):
    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])


class _too_long(object):
    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        return all([len(sentence) <= self.seq_len
                    for sentence in sentence_pair])


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
                   ('source', 'target'))
    stream = Filter(stream, predicate=_too_long(config['seq_len']))
    stream = Mapping(stream, _oov_to_unk(
                     src_vocab_size=config['src_vocab_size_%d' % i],
                     trg_vocab_size=config['trg_vocab_size'],
                     unk_id=config['unk_id']))
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       config['batch_size_enc_%d' % i]*config['sort_k_batches']))

    stream = Mapping(stream, SortMapping(_length))
    stream = Unpack(stream)
    stream = Batch(stream, iteration_scheme=ConstantScheme(
        config['batch_size_enc_%d' % i]))
    masked_stream = Padding(stream)
    masked_stream = Mapping(
        masked_stream, _remapWordIdx([(0, 0, config['src_eos_idx_%d' % i]),
                                     (2, 0, config['trg_eos_idx'])]))
    ind_streams.append(masked_stream)

multi_enc_stream = MultiEncStream(ind_streams, schedule=config['schedule'],
                                  batch_sizes=[config['batch_size_enc_%d' % i]
                                               for i in xrange(num_encs)])

# Development set streams *****************************************************
# Setup development set stream if necessary
dev_streams = []
for i in xrange(config['num_encs']):
    if 'val_set_%d' % i in config and config['val_set_%d' % i]:
        dev_file = config['val_set_%d' % i]
        dev_dataset = TextFile(
            [dev_file], cPickle.load(open(src_vocabs[i])), None)
        dev_streams.append(DataStream(dev_dataset))
