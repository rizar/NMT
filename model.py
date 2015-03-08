#
# WARNING: VERY UNTESTED
# This is an encoder-decoder model I slapped together
# It only works with my changes in Blocks PRs #414 and #423 merged
# I have no idea whether it actually trains, I haven't run more than 5 epochs
#

from theano import tensor

from blocks.algorithms import (GradientDescent, StepClipping, AdaDelta,
                               CompositeRule)
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.bricks import (Tanh, Identity, LinearMaxout, Linear,
                           Initializable, FeedforwardSequence)
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
lookup = LookupTable(30000, 512, name='english_embeddings')
linear = Linear(input_dim=512, output_dim=1000,
                weights_init=IsotropicGaussian(0.1), biases_init=Constant(0))
rnn_input = linear.apply(lookup.apply(x_t))

encoder = GatedRecurrent(Tanh(), None, 1000, name='encoder')

last_hidden_state = encoder.apply(rnn_input, rnn_input, rnn_input,
                                  mask=x_mask_t)[-1]


class InitializableFeedforwardSequence(FeedforwardSequence, Initializable):
    pass


# The decoder
readout = Readout(source_names=['states', 'feedback'],
                  readout_dim=30000,
                  emitter=SoftmaxEmitter(),
                  feedback_brick=LookupFeedback(30000, 1000),
                  merge_prototype=Identity(),
                  post_merge=InitializableFeedforwardSequence(
                      [LinearMaxout(output_dim=500, num_pieces=2).apply,
                       Linear(input_dim=500).apply]),
                  merged_dim=1000)

sequence_generator = SequenceGenerator(
    readout=readout,
    transition=GatedRecurrent(Tanh(), dim=1000, name='decoder')
)


# Calculate the cost
cost = sequence_generator.cost(outputs=y_t, mask=y_mask_t,
                               states=last_hidden_state)
cost = (cost * y_mask_t).sum() / y_mask_t.sum()
cost.name = 'cost'

# Initialization of the weights
encoder.weights_init = Orthogonal()
sequence_generator.weights_init = IsotropicGaussian(0.1)
sequence_generator.biases_init = Constant(0)
sequence_generator.push_initialization_config()
sequence_generator.transition.weights_init = Orthogonal()
lookup.weights_init = IsotropicGaussian(0.1)

encoder.initialize()
sequence_generator.initialize()
lookup.initialize()
linear.initialize()

# Set up training algorithm (standard SGD with gradient clipping)
cg = ComputationGraph(cost)
algorithm = GradientDescent(
    cost=cost, params=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), AdaDelta()])
)

# Train!
main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=masked_stream,
    extensions=[
        TrainingDataMonitoring([cost], after_every_batch=True),
        Printing(after_every_batch=True)
    ]
)
main_loop.run()
