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

# Everthing here should be wrapped and parameterized by config
# this import is to workaround for pickling errors when wrapped
from model import config


def _length(sentence_pair):
    return len(sentence_pair[1])


def _oov_to_unk(sentence_pair, src_vocab_size=30000,
                trg_vocab_size=30000, unk_id=1):
    src_vocab_size = src_vocab_size
    trg_vocab_size = trg_vocab_size
    unk_id = unk_id
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def _too_long(sentence_pair, seq_len=50):
    return all([len(sentence) < seq_len
                for sentence in sentence_pair])

fi_vocab = config['src_vocab']
en_vocab = config['trg_vocab']
fi_file = config['src_data']
en_file = config['trg_data']

fi_dataset = TextFile([fi_file], cPickle.load(open(fi_vocab)), None)
en_dataset = TextFile([en_file], cPickle.load(open(en_vocab)), None)

stream = Merge([fi_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('source', 'target'))

stream = Filter(stream, predicate=_too_long,
                predicate_args={'seq_len':config['seq_len']})
stream = Mapping(stream, _oov_to_unk,
                 src_vocab_size=config['src_vocab_size'],
                 trg_vocab_size=config['trg_vocab_size'],
                 unk_id=config['unk_id'])
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   config['batch_size']*config['sort_k_batches']))

stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(config['batch_size']))
masked_stream = Padding(stream)

# Setup development set stream if necessary
dev_stream = None
if 'val_set' in config and config['val_set']:
    dev_file = config['val_set']
    dev_dataset = TextFile([dev_file], cPickle.load(open(fi_vocab)), None)
    dev_stream = DataStream(dev_dataset)

