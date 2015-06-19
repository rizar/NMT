from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    BaseSequenceGenerator, LookupFeedback)
from blocks.utils import dict_union, dict_subset

from multiCG_attention import AttentionRecurrentWithMultiContext

from theano import tensor


class LookupFeedbackWMT15(LookupFeedback):

    @application
    def feedback(self, outputs):
        assert self.output_dim == 0

        shp = [outputs.shape[i] for i in xrange(outputs.ndim)]
        outputs_flat = outputs.flatten()
        outputs_flat_zeros = tensor.switch(outputs_flat < 0, 0,
                                           outputs_flat)

        lookup_flat = tensor.switch(
            outputs_flat[:, None] < 0,
            tensor.alloc(0., outputs_flat.shape[0], self.feedback_dim),
            self.lookup.apply(outputs_flat_zeros))
        lookup = lookup_flat.reshape(shp+[self.feedback_dim])
        return lookup


class SequenceGeneratorWithMultiContext(BaseSequenceGenerator):
    def __init__(self, num_contexts, readout, transition, attention=None,
                 add_contexts=True, **kwargs):
        normal_inputs = [name for name in transition.apply.sequences
                         if 'mask' not in name]
        kwargs.setdefault('fork', Fork(normal_inputs))
        self.num_contexts = num_contexts
        transition = AttentionRecurrentWithMultiContext(
            num_contexts, transition, attention,
            name="att_trans")
        super(SequenceGeneratorWithMultiContext, self).__init__(
            readout, transition, **kwargs)

    @application
    def get_transition_func(
            self, **kwargs):
            states = dict_subset(kwargs, self._state_names, must_have=False)
            contexts = dict_subset(kwargs, self._context_names)
            feedback = self.readout.feedback(kwargs['outputs'])
            inputs = self.fork.apply(feedback, as_dict=True)
            return self.transition.apply(
                mask=kwargs['mask'], return_initial_states=True, as_dict=True,
                **dict_union(inputs, states, contexts))
