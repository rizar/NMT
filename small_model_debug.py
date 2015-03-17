import numpy

from numpy import dot, tanh, exp

def sigmoid(x):
    return 1./(1+exp(-x))

###

emb =enc_param_dict['/encoder/embeddings.W'].get_value()

W = enc_param_dict['/encoder/fork/fork_inputs.W'].get_value()
Wz = enc_param_dict['/encoder/fork/fork_update_inputs.W'].get_value()
Wr = enc_param_dict['/encoder/fork/fork_reset_inputs.W'].get_value()

U = enc_param_dict['/encoder/encoder_transition.state_to_state'].get_value()
Uz = enc_param_dict['/encoder/encoder_transition.state_to_update'].get_value()
Ur = enc_param_dict['/encoder/encoder_transition.state_to_reset'].get_value()

###

d_I = dec_param_dict['/decoder/fork/fork_states.W'].get_value()

d_emb = dec_param_dict['/decoder/sequencegenerator/readout/lookupfeedback/lookuptable.W'].get_value()

d_W = dec_param_dict['/decoder/sequencegenerator/fork/fork_inputs.W'].get_value()
d_Wz = dec_param_dict['/decoder/sequencegenerator/fork/fork_update_inputs.W'].get_value()
d_Wr = dec_param_dict['/decoder/sequencegenerator/fork/fork_reset_inputs.W'].get_value()

d_U = dec_param_dict['/decoder/sequencegenerator/with_fake_attention/decoder/decoder.state_to_state'].get_value()
d_Uz = dec_param_dict['/decoder/sequencegenerator/with_fake_attention/decoder/decoder.state_to_update'].get_value()
d_Ur = dec_param_dict['/decoder/sequencegenerator/with_fake_attention/decoder/decoder.state_to_reset'].get_value()

d_C = dec_param_dict['/decoder/fork/fork_transition_context.W'].get_value()
d_Cz = dec_param_dict['/decoder/fork/fork_update_context.W'].get_value()
d_Cr = dec_param_dict['/decoder/fork/fork_reset_context.W'].get_value()

d_Oh = dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_states.W'].get_value()
d_Oy = dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_feedback.W'].get_value()
d_Oc = dec_param_dict['/decoder/sequencegenerator/readout/merge/transform_readout_context.W'].get_value()

d_W1 = decoder.children[1].children[0].children[3].children[2].params[0].get_value()
d_W2 = dec_param_dict['/decoder/sequencegenerator/readout/initializablefeedforwardsequence/linear.W'].get_value()

####

#Encoder sentence 1

z1 = sigmoid(dot(emb[3], Wz))
r1 = sigmoid(dot(emb[3], Wr))

g1 = tanh(dot(emb[3], W))
h1 = z1 * g1

z2 = sigmoid(dot(emb[0], Wz) + dot(h1, Uz))
r2 = sigmoid(dot(emb[0], Wr) + dot(h1, Ur))

g2 = tanh(dot(emb[0], W) + dot(r2*h1,U))
h2 = (1 - z2) * h1 + z2 * g2

c = h2

###

# Decoder sentence 1

d_h0 = tanh(dot(c, d_I))

d_z1 = sigmoid(dot(d_h0, d_Uz) + dot(c, d_Cz))
d_r1 = sigmoid(dot(d_h0, d_Ur) + dot(c, d_Cr))

d_g1 = tanh(dot(d_r1*d_h0, d_U) + dot(c, d_C))
d_h1 = (1 - d_z1) * d_h0 + d_z1 * d_g1

# THE FOLLOWING IS WRONG BUT THIS IS WHAT THE MODEL DOES
"""
d_z1 = sigmoid(dot(d_emb[2],d_Wz) + dot(d_h0, d_Uz) + dot(c, d_Cz))
d_r1 = sigmoid(dot(d_emb[2],d_Wr) + dot(d_h0, d_Ur) + dot(c, d_Cr))

d_g1 = tanh(dot(d_emb[2],d_W) + dot(d_r1*d_h0, d_U) + dot(c, d_C))
d_h1 = (1 - d_z1) * d_h0 + d_z1 * d_g1
"""
