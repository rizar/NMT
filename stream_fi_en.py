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
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from states import get_states_wmt15_fi_en_40k

state = get_states_wmt15_fi_en_40k()

fi_vocab, en_vocab = [os.path.join('/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed',
                                   'vocab.{}.pkl'.format(lang))
                      for lang in ['fi', 'en']]

fi_files = [os.path.join('/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed',
                         'all.tok.clean.shuf.seg1.fi-en.fi')]

en_files = [os.path.join('/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed',
                         'all.tok.clean.shuf.fi-en.en')]

dev_file = os.path.join('/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev',
                        'newsdev2015.tok.seg.fi')

class CycleTextFile(TextFile):
    """This dataset cycles through the text files, reading a sentence
    from each.
    """
    def open(self):
        return chain.from_iterable(izip(*[chain.from_iterable(
            imap(open, repeat(f))) for f in self.files]))

fi_dataset = CycleTextFile(fi_files, cPickle.load(open(fi_vocab)), None)
en_dataset = CycleTextFile(en_files, cPickle.load(open(en_vocab)), None)

stream = Merge([fi_dataset.get_example_stream(),
                en_dataset.get_example_stream()],
               ('finnish', 'english'))

dev_dataset = TextFile([dev_file], cPickle.load(open(fi_vocab)), None)
dev_stream = DataStream(dev_dataset)

def _oov_to_unk(sentence_pair, src_vocab_size=state['src_vocab_size'],
                trg_vocab_size=state['trg_vocab_size'], unk_id=state['unk_id']):
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])

def _too_long(sentence_pair):
    return all([len(sentence) < state['seq_len']
                for sentence in sentence_pair])

def _length(sentence_pair):
    return len(sentence_pair[1])

stream = Filter(stream, predicate=_too_long)
stream = Mapping(stream, _oov_to_unk)
stream = Batch(stream,
               iteration_scheme=ConstantScheme(
                   state['batch_size']*state['sort_k_batches']))
stream = Mapping(stream, SortMapping(_length))
stream = Unpack(stream)
stream = Batch(stream, iteration_scheme=ConstantScheme(state['batch_size']))
masked_stream = Padding(stream)
