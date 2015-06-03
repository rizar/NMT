#!/usr/bin/env python

import cPickle as pkl
import argparse
import logging
import operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('generate_decoder_sub_vocab')

parser = argparse.ArgumentParser()
parser.add_argument(
    "--small", type=str, help="Target (small) vocabulary of decoder")
parser.add_argument(
    "--large", type=str, help="Target (large) vocabulary of decoder")
parser.add_argument(
    "--out", type=str, help="Output file name, pickle")


def main(args):

    large = pkl.load(open(args.large))
    small = pkl.load(open(args.small))

    # Create subset vocabulary with intersection
    tmp_vocab = dict([(key, value) for key, value in small.iteritems()
                      if large.has_key(key) and large[key] < 200000])

    tmp_vocab = sorted(tmp_vocab.items(), key=operator.itemgetter(1))

    # Put unk, eos, bos to subset vocabulary
    # This is the conversion from Ghog to Blocks tokens
    sub_vocab = {}
    sub_vocab['<S>'] = 0
    sub_vocab['</S>'] = 0
    sub_vocab['<UNK>'] = 1
    logger.info(" setting {} to {}".format('<S>', 0))
    logger.info(" setting {} to {}".format('</S>', 0))
    logger.info(" setting {} to {}".format('<UNK>', 1))

    for i in xrange(len(tmp_vocab)):
        sub_vocab[tmp_vocab[i][0]] = 2 + i

    logger.info("Size of sub-vocab [{}]".format(len(sub_vocab.keys())))
    logger.info("Min-Max of sub-vocab [{}]-[{}]".format(
        min(sub_vocab.values()), max(sub_vocab.values())))
    logger.info(" saving sub-vocabulary to {}".format(args.out))
    pkl.dump(sub_vocab, open(args.out, 'w'))

    logger.info('Done!')
    print ""


if __name__ == "__main__":
    main(parser.parse_args())
