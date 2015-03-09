# This is the original encoder-decoder model
# It only works with Blocks PR #414 merged. It seems to train, but
# I haven't monitored validation error, checkpointed or sampled sentences
# TIP: Without CuDNN Theano seems to move part of the step clipping to CPU
#      on my computer, which makes things very slow. CuDNN gives a 2x speedup
#      in my case, so it's worth installing.

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
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter, SequenceGenerator
)

from stream import masked_stream

# Create Theano variables
x = tensor.lmatrix('english')
x_mask = tensor.matrix('english_mask')
y = tensor.lmatrix('french')
y_mask = tensor.matrix('french_mask')

# Time as first dimension
x_t = x.dimshuffle(1, 0)[::-1]
x_mask_t = x_mask.T[::-1]
y_t = y.dimshuffle(1, 0)
y_mask_t = y_mask.T

# Encoder
lookup = LookupTable(30000, 100, name='english_embeddings')
linear = Linear(input_dim=100, output_dim=1000)
rnn_input = linear.apply(lookup.apply(x_t))

encoder = GatedRecurrent(Tanh(), None, 1000, name='encoder')

last_hidden_state = encoder.apply(rnn_input, rnn_input, rnn_input,
                                  mask=x_mask_t)[-1]
context_transform = MLP(dims=[1000, 1000], activations=[Tanh()])
context = context_transform.apply(last_hidden_state)


class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass

# The decoder
readout = Readout(source_names=['states', 'feedback'],
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
    readout=readout,
    transition=GatedRecurrent(Tanh(), dim=1000, name='decoder')
)


# Calculate the cost
cost = sequence_generator.cost(outputs=y_t, mask=y_mask_t,
                               states=context)
cost = (cost * y_mask_t).sum() / y_mask_t.sum()
cost.name = 'cost'

# Initialization of the weights
lookup.weights_init = IsotropicGaussian(0.1)
linear.weights_init = IsotropicGaussian(0.1)
linear.biases_init = Constant(0)
encoder.weights_init = Orthogonal()
context_transform.weights_init = IsotropicGaussian(0.1)
context_transform.biases_init = Constant(0)
sequence_generator.weights_init = IsotropicGaussian(0.1)
sequence_generator.biases_init = Constant(0)
sequence_generator.push_initialization_config()
sequence_generator.transition.weights_init = Orthogonal()
readout.post_merge.weights_init = IsotropicGaussian(0.1)
readout.post_merge.biases_init = Constant(0)

lookup.initialize()
linear.initialize()
encoder.initialize()
context_transform.initialize()
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
