#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint
import shelve

import numpy
import logging

from collections import OrderedDict

from custom_stream_fi_en import masked_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Does not currently use the masks
    # If the padding is done with the <EOS> token,
    # or any token appearing in all batches,
    # there should be no side effect.

    # Make sure the RNG is the same
    
    src_name = 'finnish'
    trg_name = 'english'
    Dx_fname = '/data/lisatmp3/jeasebas/nmt/debug_blocks_LV/en2fi_100k_2k_Dx_file'
    Dy_fname = '/data/lisatmp3/jeasebas/nmt/debug_blocks_LV/en2fi_100k_2k_Dy_file'
    rolling_vocab_fname = '/data/lisatmp3/jeasebas/nmt/debug_blocks_LV/en2fi_100k_2k_rolling_vocab_dict.pkl'
    n_sym_target = 2000
    
    train_data = masked_stream.get_epoch_iterator(as_dict=True)

    Dx = OrderedDict()
    dy = OrderedDict()
    Dy = OrderedDict()
    Cy = OrderedDict()

    for i in xrange(n_sym_target):
        Dy[i] = i
        Cy[i] = i

    def update_dicts(arr, d, D, C, full):
        i_range, j_range = numpy.shape(arr)
        for i in xrange(i_range):
            for j in xrange(j_range):
                word = arr[i,j]
                if word not in d:
                    if len(d) == full:
                        return True
                    if word not in D: # Also not in C
                        key, value = C.popitem()
                        del D[key]
                        d[word] = value
                        D[word] = value
                    else: # Also in C as (d UNION C) is D. (d INTERSECTION C) is the empty set.
                        d[word] = D[word]
                        del C[word]
        return False

    def unlimited_update_dicts(arr, D, size):
        i_range, j_range = numpy.shape(arr)
        for i in xrange(i_range):
            for j in xrange(j_range):
                word = arr[i,j]
                if word not in D:
                    D[word] = size
                    size += 1

    prev_step = 0
    step = 0
    rolling_vocab_dict = {}
    Dx_dict = {}
    Dy_dict = {}

    output = False
    stop = False

    while not stop: # Assumes the shuffling in get_homogeneous_batch_iter is always the same (Is this true?)
        try:
            batch = next(train_data)
            if step == 0:
                rolling_vocab_dict[step] = (batch[src_name][:,0].tolist(), batch[trg_name][:,0].tolist())
        except:
            batch = None
            stop = True

        if batch:
            unlimited_update_dicts(batch[src_name], Dx, len(Dx))
            output = update_dicts(batch[trg_name], dy, Dy, Cy, n_sym_target)

            if output:
                Dx_dict[prev_step] = Dx.copy() # Save dictionaries for the batches preceding this one
                Dy_dict[prev_step] = Dy.copy()
                rolling_vocab_dict[step] = (batch[src_name][:,0].tolist(), batch[trg_name][:,0].tolist()) # When we get to this batch, we will need to use a new vocabulary
                if (step/100000 - prev_step/100000):
                    logger.info('Updating Dx, Dy')
                    Dx_file = shelve.open(Dx_fname)
                    Dy_file = shelve.open(Dy_fname)
                    for key in Dx_dict:
                        Dx_file[str(key)] = Dx_dict[key]
                        Dy_file[str(key)] = Dy_dict[key]
                    Dx_file.close()
                    Dy_file.close()
                    Dx_dict = {}
                    Dy_dict = {}
                # tuple of first sentences of the batch # Uses large vocabulary indices
                prev_step = step
                logger.info("%i %i" % (step, len(Dx)))
                Dx = OrderedDict()
                dy = OrderedDict()
                Cy = Dy.copy()
                output = False

                unlimited_update_dicts(batch[src_name], Dx, len(Dx))
                update_dicts(batch[trg_name], dy, Dy, Cy, n_sym_target)
            
            step += 1

    Dx_dict[prev_step] = Dx.copy()
    Dy_dict[prev_step] = Dy.copy()
    rolling_vocab_dict[step]=0 # Total number of batches # Don't store first sentences here

    with open(rolling_vocab_fname,'w') as f:
        cPickle.dump(rolling_vocab_dict, f)
    Dx_file = shelve.open(Dx_fname)
    Dy_file = shelve.open(Dy_fname)
    for key in Dx_dict:
        Dx_file[str(key)] = Dx_dict[key]
        Dy_file[str(key)] = Dy_dict[key]
    Dx_file.close()
    Dy_file.close()

if __name__ == "__main__":
    main()
