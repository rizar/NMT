#!/usr/bin/env python

import cPickle as pkl
import numpy as np
import argparse
import os
from config import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('replace_decoder_emb')

parser = argparse.ArgumentParser()
parser.add_argument("--large", type=str, help="Decoder large vocabulary")
parser.add_argument(
    "--small", type=str, help="Sub vocabulary created by generate_sub_vocab")
parser.add_argument("--gh-model", type=str, help="GroundHog model.npz")
parser.add_argument("--bl-model", type=str, default=None, help="Blocks model.npz")
parser.add_argument("--config", type=str, help="Configuration prototype name")
parser.add_argument(
    "--out", type=str, default="params.replaced.npz", help="Output file name")


def main(args):
    large = pkl.load(open(args.large))
    small = pkl.load(open(args.small))
    gh_model = dict(np.load(open(args.gh_model)))
    bl_model = {}
    if args.bl_model:
        bl_model = np.load(open(args.bl_model))

    # Get corresponding config
    config = eval(args.config)()
    trg_vocab_size = config['trg_vocab_size']
    trg_embed_size = config['dec_embed']

    bl_param = '-decoder-sequencegeneratorwithmulticontext-readout-lookupfeedbackwmt15-lookuptable.W'
    gh_param = 'W_0_dec_approx_embdr'
    gh_bias = 'b_0_dec_approx_embdr'

    # Get embedding param
    gh_emb_W = gh_model[gh_param]
    gh_emb_b = gh_model.get(gh_bias, None)

    if gh_emb_b is not None:
        logger.info(" setting {} to [{}] + [{}]".format(bl_param, gh_param, gh_bias))
    else:
        logger.info(" setting {} to [{}]".format(bl_param, gh_param))

    # Init embeddings
    bl_emb = np.zeros((trg_vocab_size, trg_embed_size), dtype=gh_emb_W.dtype)

    # Set embeddings
    for i, (word, idx) in enumerate(small.iteritems()):
        gh_idx = large.get(word, None)
        if gh_idx:
            bl_emb[idx, :] = gh_emb_W[gh_idx, :]
        else:
            logger.warning("  token {} does not exist in the large vocabulary".format(word))

    # Set unk and eos
    bl_emb[0, :] = gh_emb_W[0, :]
    bl_emb[1, :] = gh_emb_W[1, :]

    logger.info("Rearranged {} indices.".format(i+2))
    logger.info("New vocabulary size [{}]".format(trg_vocab_size))

    assert i == (trg_vocab_size-1),\
        "Vocabulary sizes are not matching for sub vocab!"

    # Add word embedding biases if necessary
    if gh_emb_b is not None:
        logger.info("Adding biases to word embeddings")
        bl_emb += gh_emb_b

    bl_model[bl_param] = bl_emb

    # Save
    out = {}
    if os.path.isfile(args.out):
        logger.info("Parameters file {} exists, loading...".format(args.out))
        out = dict(np.load(args.out))

    if bl_param in out:
        assert out[bl_param].shape == bl_emb.shape,\
            "embedding shapes are not matching!"

    out.update(bl_model)
    logger.info("Saving {}.".format(args.out))
    np.savez(args.out, **out)

    logger.info('Done!')
    print ""


if __name__ == "__main__":
    main(parser.parse_args())
