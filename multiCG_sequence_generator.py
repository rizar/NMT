from blocks.bricks.base import application
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    BaseSequenceGenerator, LookupFeedback)
from blocks.utils import dict_union, dict_subset

from multiCG_attention import AttentionRecurrentWithMultiContext

from theano import tensor


class LookupFeedbackWMT15(LookupFeedback):
    """Feedback extension to zero out initial feedback.

    This brick extends LookupFeedback and overwrites its feedback method in
    order to provide all zeros as initial feedback for Groundhog compatibility.
    It may not be needed at all since learning BOS token is a cleaner and
    better option in sequences.

    """

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
    """Sequence Generator that uses multiple contexts.

    The reason why we have such a generator is that the Sequence Generator
    structure in Blocks is not parametrized by its inner transition block.
    This sequence generator is only made in order to use
    AttentionRecurrentWithMultiContext.

    """
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
