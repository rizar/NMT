#
# WARNING: VERY UNTESTED
# This is an encoder-decoder model I slapped together
# It only works with my changes in Blocks PRs #414 and #423 merged
# I have no idea whether it actually trains, I haven't run more than 5 epochs
#

import theano
from theano import tensor

from blocks.algorithms import (GradientDescent, StepClipping, Scale,
                               CompositeRule)
from blocks.main_loop import MainLoop
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Orthogonal, Constant
from blocks.extensions import Printing
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.bricks import (Tanh, Identity, LinearMaxout, Sequence, Linear,
                           Initializable)
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    LookupFeedback, Readout, SoftmaxEmitter, SequenceGenerator
)

from stream import masked_stream

x = tensor.lmatrix('english')
x_mask = tensor.matrix('english_mask')
x_mask_t = x_mask.T
y = tensor.lmatrix('french')
y_mask = tensor.matrix('french_mask')
y_mask_t = y_mask.T
# Time as first dimension
x_t = x.dimshuffle(1, 0)
y_t = y.dimshuffle(1, 0)

lookup = LookupTable(30000, 512, name='english_embeddings')
linear = Linear(input_dim=512, output_dim=1000,
                weights_init=IsotropicGaussian(0.1), biases_init=Constant(0))
rnn_input = linear.apply(lookup.lookup(x_t))

encoder = GatedRecurrent(Tanh(), None, 1000, name='encoder')
print(encoder.apply.sequences)

last_hidden_state = encoder.apply(rnn_input, rnn_input, rnn_input, mask=x_mask_t)[-1]


class FeedforwardSequence(Sequence, Initializable):
    @application
    def apply(self, *args, **kwargs):
        return super(FeedforwardSequence, self).apply(*args, **kwargs)

    @property
    def input_dim(self):
        return self.children[0].input_dim

    @input_dim.setter
    def input_dim(self, value):
        self.children[0].input_dim = value

    @property
    def output_dim(self):
        return self.children[-1].output_dim

    @output_dim.setter
    def output_dim(self, value):
        self.children[-1].output_dim = value


readout = Readout(source_names=['states', 'feedback'],
                  readout_dim=30000,
                  emitter=SoftmaxEmitter(),
                  feedback_brick=LookupFeedback(30000, 1000),
                  merge_prototype=Identity(),
                  post_merge=FeedforwardSequence(
                      [LinearMaxout(output_dim=500, num_pieces=2).apply,
                       Linear(input_dim=500).apply]),
                  merged_dim=1000)

sequence_generator = SequenceGenerator(
    readout=readout,
    transition=GatedRecurrent(Tanh(), dim=1000, name='decoder')
)

cost = sequence_generator.cost(outputs=y_t, mask=y_mask_t,
                               states=last_hidden_state)
cost = cost.sum()
cost.name = 'cost'

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

cg = ComputationGraph(cost)

algorithm = GradientDescent(
    cost=cost, params=cg.parameters,
    step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)])
)


main_loop = MainLoop(
    algorithm=algorithm,
    data_stream=masked_stream,
    extensions=[
        TrainingDataMonitoring([cost], after_every_batch=True),
        Printing(after_every_batch=True)
    ]
)

main_loop.run()
