#
# WARNING: SLOW STREAM
# Should probably use caching and multiprocessing like in the tutorial
# The files are those from WMT15, the vocab files are simply the 30,000 most
# common words of the raw data
#

import cPickle

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

# Everthing here should be wrapped and parameterized by state
# this import is to workaround for pickling errors when wrapped
from model import state


def _length(sentence_pair):
    return len(sentence_pair[1])


def _oov_to_unk(sentence_pair):
    src_vocab_size = state['src_vocab_size']
    trg_vocab_size = state['trg_vocab_size']
    unk_id = state['unk_id']
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def _too_long(sentence_pair):
    return all([len(sentence) < state['seq_len']
                for sentence in sentence_pair])

fi_vocab = state['src_vocab']
en_vocab = state['trg_vocab']
fi_file = state['src_data']
en_file = state['trg_data']

fi_dataset = TextFile([fi_file], cPickle.load(open(fi_vocab)), None)
en_dataset = TextFile([en_file], cPickle.load(open(en_vocab)), None)

stream = Merge([fi_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('source', 'target'))

stream = Filter(stream, predicate=_too_long)
stream = Mapping(stream, _oov_to_unk)
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   state['batch_size']*state['sort_k_batches']))

stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(state['batch_size']))
masked_stream = Padding(stream)

# Setup development set stream if necessary
dev_stream = None
if 'val_set' in state and state['val_set']:
    dev_file = state['val_set']
    dev_dataset = TextFile([dev_file], cPickle.load(open(fi_vocab)), None)
    dev_stream = DataStream(dev_dataset)

