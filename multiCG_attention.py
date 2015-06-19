import theano

from blocks.bricks import Linear, Initializable
from blocks.bricks.attention import (ShallowEnergyComputer,
                                     AbstractAttentionRecurrent,
                                     GenericSequenceAttention)
from blocks.bricks.base import application, lazy
from blocks.bricks.parallel import Parallel, Distribute
from blocks.bricks.recurrent import recurrent

from blocks.utils import dict_union, dict_subset

from theano import tensor


class SequenceMultiContentAttention(GenericSequenceAttention, Initializable):
    """Should extend SequenceContentAttention"""

    @lazy(allocation=['match_dim'])
    def __init__(self, match_dim, attended_dims, state_transformer=None,
                 attended_transformers=None, energy_computer=None, **kwargs):

        # TODO: This is ugly, fix it
        kwargs['attended_dim'] = attended_dims[0]
        super(SequenceMultiContentAttention, self).__init__(**kwargs)
        self.match_dim = match_dim
        self.attended_dims = attended_dims
        self.state_transformer = state_transformer
        self.state_transformers = Parallel(input_names=self.state_names,
                                           prototype=state_transformer,
                                           name="state_trans")
        self.num_attended = len(attended_dims)
        if not attended_transformers:
            attended_transformers = [Linear(name="preprocess_%d" % i)
                                     for i in xrange(self.num_attended)]
        self.attended_transformers = attended_transformers

        if not energy_computer:
            energy_computer = ShallowEnergyComputer(name="energy_comp")
        self.energy_computer = energy_computer

        self.children = [self.state_transformers, energy_computer] +\
            attended_transformers

    def _push_allocation_config(self):
        self.state_transformers.input_dims = self.state_dims
        self.state_transformers.output_dims = [self.match_dim
                                               for name in self.state_names]
        for i in xrange(self.num_attended):
            self.attended_transformers[i].input_dim = self.attended_dims[i]
            self.attended_transformers[i].output_dim = self.match_dim
        self.energy_computer.input_dim = self.match_dim
        self.energy_computer.output_dim = 1

    @application
    def compute_energies(self, attendeds, preprocessed_attendeds,
                         states):
        if not all(preprocessed_attendeds):
            preprocessed_attendeds = self.preprocess(attendeds)
        transformed_states = self.state_transformers.apply(as_dict=True,
                                                           **states)

        # Broadcasting of transformed states should be done automatically
        match_vectors = transformed_states.values()
        for att in preprocessed_attendeds:
            match_vectors += att
        energies = self.energy_computer.apply(match_vectors).reshape(
            match_vectors.shape[:-1], ndim=match_vectors.ndim - 1)

        self._inner_fn_get_match_vectors = theano.function(
            [states['states']], transformed_states.values()[0])

        self._inner_fn_get_preprocessed_attendeds = theano.function(
            attendeds, self.preprocess(attendeds))

        return energies

    @application(outputs=['weighted_averages', 'weights'])
    def take_glimpses(self, attendeds, preprocessed_attendeds=None,
                      attended_mask=None, **states):
        energies = self.compute_energies(attendeds, preprocessed_attendeds,
                                         states)
        weights = self.compute_weights(energies, attended_mask)
        weighted_averages = self.compute_weighted_averages(
            weights, attendeds[0])
        return weighted_averages, weights.T

    @take_glimpses.property('inputs')
    def take_glimpses_inputs(self):
        return (
            ['attended_%d' % i for i in xrange(self.num_attended)] +
            ['preprocessed_attended_%d' % i
             for i in xrange(self.num_attended)] +
            ['attended_mask'] + self.state_names)

    @application
    def initial_glimpses(self, name, batch_size, attended):
        # TODO: adapted to return initial states for both weighted_averages and
        # weights at the same time for both calls to ensure different output
        # names, NOTE that: ordering matters
        if name == "weighted_averages" or name == "weights":
            return [tensor.zeros((batch_size, self.attended_dims[0])),
                    tensor.zeros((batch_size, attended[0].shape[0]))]
        raise ValueError("Unknown glimpse name {}".format(name))

    @application(inputs=['attended'],
                 outputs=['preprocessed_attended_0',
                          'preprocessed_attended_1',
                          'preprocessed_attended_2'])
    def preprocess(self, attended):
        preprocessed_attended = []
        for i, att in enumerate(attended):
            preprocessed_attended.append(
                self.attended_transformers[i].apply(att))
        return preprocessed_attended

    def get_dim(self, name):
        if name in ['weighted_averages']:
            return self.attended_dims[0]
        if name in ['weights']:
            return 0
        if name in ['attended_%d' % i
                    for i in xrange(self.num_attended)]:
            return self.attended_dims[int(name[-1])]
        if name in ['preprocessed_attended_%d' % i
                    for i in xrange(self.num_attended)]:
            return self.match_dim
        return super(SequenceMultiContentAttention, self).get_dim(name)


class AttentionRecurrentWithMultiContext(AbstractAttentionRecurrent,
                                         Initializable):

    def __init__(self, num_contexts, transition, attention, **kwargs):
        super(AttentionRecurrentWithMultiContext, self).__init__(**kwargs)
        self._sequence_names = list(transition.apply.sequences)
        self._state_names = list(transition.apply.states)
        self._context_names = list(transition.apply.contexts)

        # This part is tricky
        self.num_contexts = num_contexts
        attended_names = ['attended_%d' % i for i in xrange(num_contexts)]
        attended_mask_name = 'attended_mask'

        # Construct contexts names and Remove duplicates
        self._context_names += attended_names + [attended_mask_name]
        self._context_names = list(set(self._context_names))

        normal_inputs = [name for name in self._sequence_names
                         if 'mask' not in name]
        distribute = Distribute(normal_inputs,
                                attention.take_glimpses.outputs[0])

        self.transition = transition
        self.attention = attention
        self.distribute = distribute
        self.attended_names = attended_names
        self.attended_mask_name = attended_mask_name

        self.preprocessed_attended_names = ["preprocessed_" + attended_names[i]
                                            for i in xrange(num_contexts)]

        self._glimpse_names = self.attention.take_glimpses.outputs
        # We need to determine which glimpses are fed back.
        # Currently we extract it from `take_glimpses` signature.
        self.previous_glimpses_needed = [
            name for name in self._glimpse_names
            if name in self.attention.take_glimpses.inputs]

        self.children = [self.transition, self.attention, self.distribute]

    def _push_allocation_config(self):
        self.attention.state_dims = self.transition.get_dims(
            self.attention.state_names)

        self.distribute.source_dim = self.attention.get_dim(
            self.distribute.source_name)
        self.distribute.target_dims = self.transition.get_dims(
            self.distribute.target_names)

    @application
    def take_glimpses(self, **kwargs):
        """Wrapper for attention.take_glimpses"""
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)
        glimpses_needed = dict_subset(glimpses, self.previous_glimpses_needed)
        result = self.attention.take_glimpses(
            [kwargs.pop(name) for name in self.attended_names],
            [kwargs.pop(name, None) for name in
                self.preprocessed_attended_names],
            kwargs.pop(self.attended_mask_name, None),
            **dict_union(states, glimpses_needed))
        if kwargs:
            raise ValueError("extra args to take_glimpses: {}".format(kwargs))
        return result

    @take_glimpses.property('outputs')
    def take_glimpses_outputs(self):
        return self._glimpse_names

    @application
    def compute_states(self, **kwargs):
        # Masks are not mandatory, that's why 'must_have=False'
        sequences = dict_subset(kwargs, self._sequence_names,
                                pop=True, must_have=False)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

        # This is the additional context to GRU from source selector
        contexts = dict_subset(kwargs, self.transition.apply.contexts,
                               pop=False)

        for name in self.attended_names:
            kwargs.pop(name)
        kwargs.pop(self.attended_mask_name)

        sequences.update(self.distribute.apply(
            as_dict=True, **dict_subset(dict_union(sequences, glimpses),
                                        self.distribute.apply.inputs)))

        current_states = self.transition.apply(
            iterate=False, as_list=True,
            **dict_union(sequences, contexts, kwargs))
        return current_states

    @compute_states.property('outputs')
    def compute_states_outputs(self):
        return self._state_names

    @recurrent
    def do_apply(self, **kwargs):
        attendeds_dict = {}
        preprocessed_attendeds_dict = {}
        # ordering is important
        for i in xrange(self.num_contexts):
            att_name = self.attended_names[i]
            p_att_name = self.preprocessed_attended_names[i]
            attendeds_dict[att_name] = kwargs[att_name]
            preprocessed_attendeds_dict[p_att_name] = kwargs.pop(p_att_name)

        attended_mask = kwargs.get(self.attended_mask_name)
        sequences = dict_subset(kwargs, self._sequence_names, pop=True,
                                must_have=False)
        states = dict_subset(kwargs, self._state_names, pop=True)
        glimpses = dict_subset(kwargs, self._glimpse_names, pop=True)

        current_glimpses = self.take_glimpses(
            as_dict=True,
            **dict_union(
                states, glimpses, attendeds_dict,
                preprocessed_attendeds_dict,
                {self.attended_mask_name: attended_mask}))

        current_states = self.compute_states(
            as_list=True,
            **dict_union(sequences, states, current_glimpses, kwargs))
        return current_states + list(current_glimpses.values())

    @do_apply.property('sequences')
    def do_apply_sequences(self):
        return self._sequence_names

    @do_apply.property('contexts')
    def do_apply_contexts(self):
        return self._context_names + self.preprocessed_attended_names

    @do_apply.property('states')
    def do_apply_states(self):
        return self._state_names + self._glimpse_names

    @do_apply.property('outputs')
    def do_apply_outputs(self):
        return self._state_names + self._glimpse_names

    @application
    def apply(self, **kwargs):
        preprocessed_attended = self.attention.preprocess(
            [kwargs[name] for name in self.attended_names])
        add_kwargs = dict(zip(self.preprocessed_attended_names,
                              preprocessed_attended))
        new_kwargs = dict_union(kwargs, add_kwargs)
        return self.do_apply(**new_kwargs)

    @apply.delegate
    def apply_delegate(self):
        # TODO: Nice interface for this trick?
        # This was originally in blocks, check recent version
        return self.do_apply.__get__(self, None)

    @apply.property('contexts')
    def apply_contexts(self):
        return self._context_names

    @application
    def initial_state(self, state_name, batch_size, **kwargs):
        if state_name in self._glimpse_names:
            # TODO: find a better solution to this, variable name for both
            # weigted_averages and weights returns as the same,
            # 'attention_initial_glimpses_output_0' which should be different
            return self.attention.initial_glimpses(
                state_name, batch_size, [kwargs[name] for name in
                                         self.attended_names]
                )[self._glimpse_names.index(state_name)]
        return self.transition.initial_state(state_name, batch_size, **kwargs)

    def get_dim(self, name):
        if name in self._glimpse_names:
            return self.attention.get_dim(name)
        if name in self.preprocessed_attended_names:
            (original_name,) = self.attention.preprocess.outputs
            return self.attention.get_dim(original_name)
        # TODO: this is a bit tricky, find a better soln
        # Since we use multiple attendeds, each attended is identified by its
        # suffix which is basically an integer index, this will crash otherwise
        if name in self.attended_names:
            return self.attention.get_dim(
                self.attention.take_glimpses.inputs[int(name[-1])])
        if name == self.attended_mask_name:
            return 0
        return self.transition.get_dim(name)
