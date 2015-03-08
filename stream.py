#
# WARNING: SLOW STREAM
# Should probably use caching and multiprocessing like in the tutorial
# The files are those from WMT15, the vocab files are simply the 30,000 most
# common words of the raw data
#

import cPickle
import os

from fuel import config
from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.transformers import Merge, Batch, Filter, Mapping, SortMapping, Cache, Padding

en_vocab, fr_vocab = [os.path.join(config.data_path, 'mt',
                                   '{}_vocab_30000.pkl'.format(lang))
                      for lang in ['en', 'fr']]

en_sources = ['commoncrawl.fr-en.en',
              'news-commentary-v10.fr-en.en',
              'giga-fren.release2.en',
              'training/europarl-v7.fr-en.en',
              'un/undoc.2000.fr-en.en']
en_files = [os.path.join(config.data_path, 'mt', source)
            for source in en_sources]

fr_sources = ['commoncrawl.fr-en.fr',
              'news-commentary-v10.fr-en.fr',
              'giga-fren.release2.fr',
              'training/europarl-v7.fr-en.fr',
              'un/undoc.2000.fr-en.fr']
fr_files = [os.path.join(config.data_path, 'mt', source)
            for source in fr_sources]

en_dataset = TextFile(en_files, cPickle.load(open(en_vocab)), None, None)
fr_dataset = TextFile(fr_files, cPickle.load(open(fr_vocab)), None)

stream = Merge([en_dataset.get_example_stream(),
                fr_dataset.get_example_stream()],
               ('english', 'french'))


def too_long(sentence_pair):
    return all([len(sentence) < 50 for sentence in sentence_pair])

filtered_stream = Filter(stream, predicate=too_long)
batched_stream = Batch(filtered_stream, iteration_scheme=ConstantScheme(32))
masked_stream = Padding(batched_stream)
