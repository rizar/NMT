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
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import Merge, Batch, Filter, Padding, Mapping

from config import get_config_wmt15_fi_en_40k

config = get_config_wmt15_fi_en_40k()

en_vocab, fr_vocab = [os.path.join(config.data_path, 'mt',
                                   'vocab.{}.pkl'.format(lang))
                      for lang in ['en', 'fr']]

en_files = [os.path.join(config.data_path, 'mt',
                         'all.tok.clean.fr-en.en')]

fr_files = [os.path.join(config.data_path, 'mt',
                         'all.tok.clean.fr-en.fr')]

dev_file = os.path.join(config.data_path, 'mt', 'dev.tok.clean.fr-en.en')


def _oov_to_unk(sentence_pair, src_vocab_size=config['src_vocab_size'],
               trg_vocab_size=config['trg_vocab_size'], unk_id=1):
    return ([x if x < src_vocab_size else unk_id for x in sentence_pair[0]],
            [x if x < trg_vocab_size else unk_id for x in sentence_pair[1]])


def too_long(sentence_pair):
    return all([len(sentence) < config['seq_len'] for sentence in sentence_pair])


class CycleTextFile(TextFile):
    """This dataset cycles through the text files, reading a sentence
    from each.
    """
    def open(self):
        return chain.from_iterable(izip(*[chain.from_iterable(
            imap(open, repeat(f))) for f in self.files]))

en_dataset = CycleTextFile(en_files, cPickle.load(open(en_vocab)), None)
fr_dataset = CycleTextFile(fr_files, cPickle.load(open(fr_vocab)), None)

stream = Merge([en_dataset.get_example_stream(),
                fr_dataset.get_example_stream()],
               ('english', 'french'))

dev_dataset = TextFile([dev_file], cPickle.load(open(en_vocab)), None)
dev_stream = DataStream(dev_dataset)

filtered_stream = Filter(stream, predicate=too_long)
filtered_stream = Mapping(filtered_stream, _oov_to_unk)
batched_stream = Batch(filtered_stream,
        iteration_scheme=ConstantScheme(config['batch_size']))
masked_stream = Padding(batched_stream)
