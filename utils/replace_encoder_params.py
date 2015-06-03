#!/usr/bin/env python

import numpy as np
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('replace_encoder_params')

"""Additional parameters in blocks encoders
[
 '-multiencoder-bidirectionalencoder_0-fwd_fork-fork_update_inputs.b',
 '-multiencoder-bidirectionalencoder_0-back_fork-fork_update_inputs.b',
 '-multiencoder-bidirectionalencoder_0-back_fork-fork_reset_inputs.b',
 '-multiencoder-bidirectionalencoder_0-fwd_fork-fork_reset_inputs.b',
]
"""

enc_params = {
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-backward.state_to_state': 'W_back_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-backward.state_to_reset': 'R_back_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-backward.state_to_update': 'G_back_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-forward.state_to_state': 'W_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-forward.state_to_reset': 'R_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-bidirectionalwmt15-forward.state_to_update': 'G_enc_transition_0',
    '-multiencoder-bidirectionalencoder_%d-back_fork-fork_reset_inputs.W': 'W_0_back_enc_reset_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-back_fork-fork_inputs.W': 'W_0_back_enc_input_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-back_fork-fork_inputs.b': 'b_0_back_enc_input_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-back_fork-fork_update_inputs.W': 'W_0_back_enc_update_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-fwd_fork-fork_reset_inputs.W': 'W_0_enc_reset_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-fwd_fork-fork_update_inputs.W': 'W_0_enc_update_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-fwd_fork-fork_inputs.W': 'W_0_enc_input_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-fwd_fork-fork_inputs.b': 'b_0_enc_input_embdr_0',
    '-multiencoder-bidirectionalencoder_%d-embeddings.W': ('W_0_enc_approx_embdr', 'b_0_enc_approx_embdr')
}

parser = argparse.ArgumentParser()
parser.add_argument("--gh-model", type=str, help="GroundHog model path")
parser.add_argument("--enc-id", type=int, help="Encoder id for replacement")
parser.add_argument("--bl-model", type=str, default=None, help="Blocks model path")
parser.add_argument("--out", type=str, default='params.replaced.npz', help="Output name")


def main(args):

    gh_model = np.load(args.gh_model)
    bl_model = {}
    if args.bl_model:
        bl_model = np.load(args.bl_model)

    # Put encoder id to parameter names
    keys = enc_params.keys()
    for bl in keys:
        enc_params[bl % args.enc_id] = enc_params.pop(bl)

    # Set parameters of blocks model
    for bl_name, gh_name in enc_params.iteritems():

        bias_name = None
        if isinstance(gh_name, tuple):
            bias_name = gh_name[1]
            gh_name = gh_name[0]

        if bias_name is not None and bias_name not in gh_model:
            logger.warn("{}: does not exist in GroundHog model".format(bias_name))
            bias_name = None

        gh_param = gh_model[gh_name]

        if bl_name in bl_model:
            assert gh_param.shape == bl_model[bl_name].shape,\
                "Parameter shape mismatch:{}".format(gh_name)
        if bias_name:
            bl_model[bl_name] = gh_param + gh_model[bias_name]
            logger.info("setting [{}] to [{}] + [{}]".format(bl_name, gh_name, bias_name))
        else:
            bl_model[bl_name] = gh_param
            logger.info("setting [{}] to [{}]".format(bl_name, gh_name))

    # Save
    model_to_save = bl_model
    if os.path.isfile(args.out):
        logger.info("Parameters file {} exists, loading...")
        old_model = dict(np.load(args.out))
        old_model.update(model_to_save)
        model_to_save = old_model

    logger.info("Saving {}.".format(args.out))
    np.savez(args.out, **model_to_save)

    logger.info("Done encoder parameter replacement!")
    print ""

if __name__ == "__main__":
    main(parser.parse_args())
