from blocks.bricks import (Tanh, Linear,
                           Initializable, MLP, Sigmoid)
from blocks.bricks.base import application
from blocks.bricks.recurrent import recurrent, Bidirectional, BaseRecurrent
from blocks.utils import shared_floatx_nans

from picklable_itertools.extras import equizip
from theano import tensor


class BidirectionalWMT15(Bidirectional):
    """Wrapper to use two RNNs with separate word embedding matrices."""

    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""
        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in
                    self.children[1].apply(reverse=True, as_list=True,
                                           **backward_dict)]
        return [tensor.concatenate([f, b], axis=2)
                for f, b in equizip(forward, backward)]


class GRUwithContext(BaseRecurrent, Initializable):
    """Gated Recurrent Unit that conditions on multiple contexts.

    GRU that conditions not only input but also source selector.
    Source selector is separately embedded for input, reset and update gates.

    Parameters
    ----------
    attended_dim : int
        The reprentation dimension of state below (encoder).
    dim : int
        The dimension of the hidden state.
    context_dim : int
        The dimension of source selector, also equal to the number of encoders
        if multiple encoders are employed.
    activation : :class:`.Brick`
        The brick to apply as activation.

    Notes
    -----
    Initial state conditions on a concatenated representation of source
    selector and last hidden state of the encoders backward rnn. An MLP with
    tanh activation is applied to the concatenated representation to obtain
    initial state of GRU.

    TODO: Computation of attended embedders should be carried outside of scan
          step function for speed up.

    """
    def __init__(self, attended_dim, dim, context_dim, activation=None,
                 gate_activation=None, use_update_gate=True,
                 use_reset_gate=True, **kwargs):
        super(GRUwithContext, self).__init__(**kwargs)
        self.dim = dim
        self.use_update_gate = use_update_gate
        self.use_reset_gate = use_reset_gate

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Sigmoid()
        self.activation = activation
        self.gate_activation = gate_activation

        self.children = [activation, gate_activation]

        self.attended_dim = attended_dim
        self.context_dim = context_dim

        # Transformer for initial state
        self.initial_transformer = MLP(
                activations=[Tanh()],
                dims=[attended_dim + context_dim, self.dim],
                name='state_initializer')
        self.children.append(self.initial_transformer)

        # Gate transformers for source selector
        self.src_selector_embedder = Linear(
                input_dim=context_dim,
                output_dim=self.dim,
                use_bias=False,
                name='src_selector_embedder')
        self.children.append(self.src_selector_embedder)
        self.src_selector_embedder_update = Linear(
                input_dim=context_dim,
                output_dim=self.dim,
                use_bias=False,
                name='src_selector_embedder')
        self.children.append(self.src_selector_embedder_update)
        self.src_selector_embedder_reset = Linear(
                input_dim=context_dim,
                output_dim=self.dim,
                use_bias=False,
                name='src_selector_embedder')
        self.children.append(self.src_selector_embedder_reset)

    @property
    def state_to_state(self):
        return self.params[0]

    @property
    def state_to_update(self):
        return self.params[1]

    @property
    def state_to_reset(self):
        return self.params[2]

    def get_dim(self, name):
        if name == 'mask':
            return 0
        if name in self.apply.sequences + self.apply.states:
            return self.dim
        if name in self.apply.contexts:
            return self.context_dim
        return super(GRUwithContext, self).get_dim(name)

    def _allocate(self):
        def new_param(name):
            return shared_floatx_nans((self.dim, self.dim), name=name)

        self.params.append(new_param('state_to_state'))
        self.params.append(new_param('state_to_update')
                           if self.use_update_gate else None)
        self.params.append(new_param('state_to_reset')
                           if self.use_reset_gate else None)

    def _initialize(self):
        self.weights_init.initialize(self.state_to_state, self.rng)
        if self.use_update_gate:
            self.weights_init.initialize(self.state_to_update, self.rng)
        if self.use_reset_gate:
            self.weights_init.initialize(self.state_to_reset, self.rng)

    @recurrent(states=['states'], outputs=['states'], contexts=['attended_1'])
    def apply(self, inputs, update_inputs=None, reset_inputs=None,
              states=None, mask=None, attended_1=None):
        if (self.use_update_gate != (update_inputs is not None)) or \
                (self.use_reset_gate != (reset_inputs is not None)):
            raise ValueError("Configuration and input mismatch: You should "
                             "provide inputs for gates if and only if the "
                             "gates are on.")

        states_reset = states

        if self.use_reset_gate:
            # TODO: move this computation out
            src_embed_reset = self.src_selector_embedder_reset.apply(
                attended_1)
            reset_values = self.gate_activation.apply(
                states.dot(self.state_to_reset) +
                reset_inputs + src_embed_reset)
            states_reset = states * reset_values

        # TODO: move this computation out
        src_embed = self.src_selector_embedder.apply(attended_1)
        next_states = self.activation.apply(
            states_reset.dot(self.state_to_state) +
            inputs + src_embed)

        if self.use_update_gate:
            # TODO: move this computation out
            src_embed_update = self.src_selector_embedder_update.apply(
                attended_1)
            update_values = self.gate_activation.apply(
                states.dot(self.state_to_update) +
                update_inputs + src_embed_update)
            next_states = (next_states * update_values +
                           states * (1 - update_values))

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)

        return next_states

    @application
    def initial_state(self, state_name, batch_size, *args, **kwargs):
        """Conditions on last hidden state and source selector."""
        if state_name == 'states':
            attended_0 = kwargs['attended_0']
            attended_1 = kwargs['attended_1']
            attended = tensor.concatenate(
                [attended_1, attended_0[0, :, -self.attended_dim:]],
                axis=1)
            initial_state = self.initial_transformer.apply(attended)
            return initial_state
        dim = self.get_dim(state_name)
        if dim == 0:
            return tensor.zeros((batch_size,))
        return tensor.zeros((batch_size, dim))

    @apply.property('sequences')
    def apply_inputs(self):
        sequences = ['mask', 'inputs']
        if self.use_update_gate:
            sequences.append('update_inputs')
        if self.use_reset_gate:
            sequences.append('reset_inputs')
        return sequences

    @apply.property('contexts')
    def apply_contexts(self):
        return ['attended_1']
