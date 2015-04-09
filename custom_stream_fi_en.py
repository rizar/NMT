#
# WARNING: SLOW STREAM
# Should probably use caching and multiprocessing like in the tutorial
# The files are those from WMT15, the vocab files are simply the 30,000 most
# common words of the raw data
#

import cPickle
import os

from picklable_itertools import chain, izip, imap, repeat

from fuel import config
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme, SequentialScheme
from fuel.transformers import (Merge, Batch, Filter,
        Padding, SortMapping, Unpack, Mapping)

# These should be set properly, Related with Issue#7
BATCH_SIZE = 80
SORT_K_BATCHES = 20
SEQ_LEN = 50
SRC_VOCAB_SIZE = 100000
TRG_VOCAB_SIZE = 100000
UNK_ID = 1

#May need to change these
fi_vocab, en_vocab = [os.path.join('/data/lisatmp3/jeasebas/nmt/debug_blocks_LV',
                                   '{}.blocks.vocab.pkl'.format(lang))
                      for lang in ['fi', 'en']]

# No segmentation for debuggging
fi_files = [os.path.join('/data/lisatmp3/jeasebas/nmt/debug_blocks_LV',
                         'all.tok.clean.shuf.fi-en.fi')]


en_files = [os.path.join('/data/lisatmp3/jeasebas/nmt/debug_blocks_LV',
                         'all.tok.clean.shuf.fi-en.en')]

fi_dataset = TextFile(fi_files, cPickle.load(open(fi_vocab)), None)
en_dataset = TextFile(en_files, cPickle.load(open(en_vocab)), None)

stream = Merge([fi_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('finnish', 'english'))


def _too_long(sentence_pair):
    return all([len(sentence) < SEQ_LEN for sentence in sentence_pair])


def _length(sentence_pair):
    return len(sentence_pair[1])


def _oov_to_unk(sentence_pair, src_vocab_size=SRC_VOCAB_SIZE,
               trg_vocab_size=TRG_VOCAB_SIZE, unk_id=UNK_ID):
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])

stream = Filter(stream, predicate=_too_long)
stream = Mapping(stream, _oov_to_unk)
stream = Batch(stream,
               iteration_scheme=ConstantScheme(BATCH_SIZE*SORT_K_BATCHES))
stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(BATCH_SIZE))
masked_stream = Padding(stream)
