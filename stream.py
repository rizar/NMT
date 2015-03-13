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
from fuel.transformers import Merge, Batch, Filter, Padding

en_vocab, fr_vocab = [os.path.join(config.data_path, 'mt',
                                   '{}_vocab_clean_30000.pkl'.format(lang))
                      for lang in ['en', 'fr']]

sources = ['commoncrawl.fr-en',
           'news-commentary-v10.fr-en',
           'giga-fren.release2',
           'training/europarl-v7.fr-en',
           'un/undoc.2000.fr-en']

en_files = [os.path.join(config.data_path, 'mt',
                         source + '.token.true.clean.en')
            for source in sources]

fr_files = [os.path.join(config.data_path, 'mt',
                         source + '.token.true.clean.fr')
            for source in sources]


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


def too_long(sentence_pair):
    return all([len(sentence) < 50 for sentence in sentence_pair])

filtered_stream = Filter(stream, predicate=too_long)
batched_stream = Batch(filtered_stream, iteration_scheme=ConstantScheme(64))
masked_stream = Padding(batched_stream)
