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

en_dataset = TextFile(['data/ntst1213.en_0'], cPickle.load(open('data/vocab.30k.en.pkl', 'rb')), None)
fr_dataset = TextFile(['data/ntst1213.fr_0'], cPickle.load(open('data/vocab.30k.fr.pkl', 'rb')), None)

stream = Merge([en_dataset.get_example_stream(),
                fr_dataset.get_example_stream()],
               ('english', 'french'))

def too_long(sentence_pair):
    return all([len(sentence) < 50 for sentence in sentence_pair])

batched_stream = Batch(stream, iteration_scheme=ConstantScheme(64))
masked_stream = Padding(batched_stream)
