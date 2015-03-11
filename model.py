# This is the original encoder-decoder model
# It only works with Blocks PR #414 merged. It seems to train, but
# I haven't monitored validation error, checkpointed or sampled sentences
# TIP: Without CuDNN Theano seems to move part of the step clipping to CPU
#      on my computer, which makes things very slow. CuDNN gives a 2x speedup
#      in my case, so it's worth installing.
import numpy
import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.plot import Plot

from blocks.bricks import (Tanh, Maxout, Linear, FeedforwardSequence, MLP,
                           Initializable)
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter, SequenceGenerator
)

from stream import masked_stream


# Helper class
class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass

# Create Theano variables
x = tensor.lmatrix('english')
x_mask = tensor.matrix('english_mask')
y = tensor.lmatrix('french')
y_mask = tensor.matrix('french_mask')

# Test values
theano.config.compute_test_value = 'warn'
x.tag.test_value = numpy.random.randint(10, size=(10, 10))
y.tag.test_value = numpy.random.randint(10, size=(10, 10))
x_mask.tag.test_value = numpy.random.rand(10, 10).astype('float32')
y_mask.tag.test_value = numpy.random.rand(10, 10).astype('float32')

# Time as first dimension
x_t = x.dimshuffle(1, 0)[::-1]
x_mask_t = x_mask.T[::-1]
y_t = y.dimshuffle(1, 0)
y_mask_t = y_mask.T

# Inputs to the model
lookup = LookupTable(30000, 100, name='english_embeddings')
fork = Fork(output_names=['linear', 'u_linear', 'r_linear'], input_dim=100,
            output_dims=dict(linear=1000, u_linear=1000, r_linear=1000))
fork.children[0].use_bias = True
rnn_input, update_rnn_input, reset_rnn_input = fork.apply(lookup.apply(x_t))

# Encoder
encoder = GatedRecurrent(Tanh(), None, 1000, name='encoder')
last_hidden_state = encoder.apply(rnn_input, update_rnn_input,
                                  reset_rnn_input, mask=x_mask_t)[-1]

# Links from the encoder to the decoder
output_to_init = MLP(dims=[1000, 1000], activations=[Tanh()])
output_fork = Fork(output_names=['to_transition', 'to_update', 'to_reset'],
                   input_dim=1000, output_dims=dict(to_transition=1000,
                   to_update=1000, to_reset=1000))
output_fork.children[0].use_bias = True

decoder_init = output_to_init.apply(last_hidden_state)
transition_context, update_context, reset_context = \
    [var.dimshuffle('x', 0, 1) for var in output_fork.apply(last_hidden_state)]
readout_context = last_hidden_state.dimshuffle('x', 0, 1)


# Decoder
class GatedRecurrentWithContext(Initializable):
    def __init__(self, *args, **kwargs):
        self.gated_recurrent = GatedRecurrent(*args, **kwargs)
        self.children = [self.gated_recurrent]

    @application(states=['states'], outputs=['states'],
                 contexts=['readout_context', 'transition_context',
                           'update_context', 'reset_context'])
    def apply(self, transition_context, update_context, reset_context,
              *args, **kwargs):
        kwargs['inputs'] += transition_context
        kwargs['update_inputs'] += update_context
        kwargs['reset_inputs'] += reset_context
        kwargs.pop('readout_context')
        return self.gated_recurrent.apply(*args, **kwargs)

    def get_dim(self, name):
        if name in ['readout_context', 'transition_context',
                    'update_context', 'reset_context']:
            return self.dim
        return self.gated_recurrent.get_dim(name)

    def __getattr__(self, name):
        return getattr(self.gated_recurrent, name)

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inputs']
        if self.use_update_gate:
            sequences.append('update_inputs')
        if self.use_reset_gate:
            sequences.append('reset_inputs')
        return sequences


# The decoder
readout = Readout(source_names=['states', 'feedback', 'readout_context'],
                  readout_dim=30000,
                  emitter=SoftmaxEmitter(),
                  feedback_brick=LookupFeedback(30000, 100),
                  post_merge=InitializableFeedforwardSequence(
                      [Maxout(num_pieces=2).apply,
                       Linear(input_dim=500, output_dim=100,
                              use_bias=False).apply,
                       Linear(input_dim=100).apply]),
                  merged_dim=1000)

sequence_generator = SequenceGenerator(
    readout=readout, fork_inputs=['inputs', 'reset_inputs', 'update_inputs'],
    transition=GatedRecurrentWithContext(Tanh(), dim=1000, name='decoder')
)
for brick in readout.merge.children:
    brick.use_bias = True
    break


# Calculate the cost
cost = sequence_generator.cost(outputs=y_t, mask=y_mask_t, states=decoder_init,
                               transition_context=transition_context,
                               update_context=update_context,
                               reset_context=reset_context,
                               readout_context=readout_context)
cost = (cost * y_mask_t).sum() / y_mask_t.sum()
cost.name = 'cost'

# Initialization of the weights
lookup.weights_init = IsotropicGaussian(0.1)
fork.weights_init = IsotropicGaussian(0.1)
fork.biases_init = Constant(0)
encoder.weights_init = Orthogonal()
output_to_init.weights_init = IsotropicGaussian(0.1)
output_to_init.biases_init = Constant(0)
output_fork.weights_init = IsotropicGaussian(0.1)
output_fork.biases_init = Constant(0)
sequence_generator.weights_init = IsotropicGaussian(0.1)
sequence_generator.biases_init = Constant(0)
sequence_generator.push_initialization_config()
sequence_generator.transition.weights_init = Orthogonal()
readout.post_merge.weights_init = IsotropicGaussian(0.1)
readout.post_merge.biases_init = Constant(0)

lookup.initialize()
fork.initialize()
encoder.initialize()
output_to_init.initialize()
output_fork.initialize()
sequence_generator.initialize()

# Set up training algorithm (standard SGD with gradient clipping)
cg = ComputationGraph(cost)
algorithm = GradientDescent(
    cost=cost, params=cg.parameters,
    step_rule=CompositeRule([StepClipping(10), AdaDelta()])
)

# Train!
main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=masked_stream,
    extensions=[
        TrainingDataMonitoring([cost], after_every_batch=True),
        Plot('En-Fr', channels=[['cost']], after_every_batch=True),
        Printing(after_every_batch=True),
        SerializeMainLoop('model.pkl', every_n_batches=2048)
    ]
)
main_loop.run()
