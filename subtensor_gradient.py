# Fix for https://github.com/Theano/Theano/issues/2219 (see also https://github.com/mila-udem/blocks/issues/672)
from collections import OrderedDict
import itertools
import logging
import numpy
import theano
from theano import tensor
from picklable_itertools.extras import equizip

from blocks.algorithms import GradientDescent, AdaDelta, Scale
from blocks.filter import VariableFilter
from blocks.bricks.lookup import LookupTable
from blocks.utils import named_copy, shared_floatx
from blocks.theano_expressions import l2_norm


logger = logging.getLogger(__name__)

def subtensor_params(cg, lookups):
    """Extract information used by the subtensor fix
    
    Parameters
    ----------
    cg : ComputationGraph
        The ComputationGraph of the model.

    lookups : list of LookupTable
        The LookupTable to optimize.
    
    Returns
    -------
    subtensor_params : dict
        Dictionary of the form {parameter: (subparam, canonized_indices, outputs, indices)}
        Where :
            - subparam is the subtensor of the parameter contributing to the gradient
            - canonized_indices is the concatenation of indices without repetition such that param[canonized_indices] = subparam
            - outputs is the list of subtensors in the graph which are result of a lookup
            - indices is the list of indices in the graph which are used in a lookup so that forall i: param[indices[i]] = outputs[i]
    """

    def extract_ind_app(lookup):
        assert isinstance(lookup, LookupTable)
        param = lookup.W
        branches = VariableFilter(bricks=[lookup], name='output_0')(cg)
        assert all([
            isinstance(branch.owner.inputs[0].owner.inputs[0].owner.op,
                       tensor.subtensor.AdvancedSubtensor1)
            for branch in branches])
        outputs = [branch.owner.inputs[0].owner.inputs[0] for branch in branches]
        indices = [branch.owner.inputs[0].owner.inputs[0].owner.inputs[1] for branch in branches]
        canonized_indices = tensor.concatenate(indices, axis=0)
        canonized_indices = canonized_indices % lookup.length # Replace -1 with lookup.length - 1
        canonized_indices = tensor.extra_ops.Unique()(tensor.sort(canonized_indices))

        subparam = param[canonized_indices]
        return {param: (subparam, canonized_indices, outputs, indices)}

    r = {}
    for lookup in lookups:
        r.update(extract_ind_app(lookup))
    return r

class GradientDescent_SubtensorFix(GradientDescent):
    """Gradient descent with support for indexed gradients.

    Parameters
    ----------
    subtensor_params : dict
        A dictionary given by the subtensor_params function.

    """
    def __init__(self, cost, params, subtensor_params={}, step_rule=None, *args, **kwargs):
        full_params = params
        self.subtensor_params = subtensor_params

        # For each LookupTable, we replace it by its subtensors appearing in the graph
        params = [param for param in full_params if param not in subtensor_params]
        for _, (_, _, outputs, _) in subtensor_params.iteritems():
            params.extend(outputs)

        super(GradientDescent, self).__init__(cost=cost, params=params, **kwargs)
        # self.params contains the list of outputs of the lookup tables

        logger.info("Taking the cost gradient")
        self.gradients = dict(
            equizip(self.params, tensor.grad(self.cost, self.params)))

        # We combine the gradients extracted from the same parameter
        for param, (subparam, canonized_indices, outputs, indices) in subtensor_params.iteritems():
            # This is necessary if we want to compute the l2 norm correctly (e.g. for StepClipping)
            tmp = shared_floatx(param.get_value() * 0.)
            for (output, indice) in zip(outputs, indices):
                tmp = tensor.inc_subtensor(tmp[indice], self.gradients[output])
                del self.gradients[output]
            self.gradients[subparam] = tmp[canonized_indices]

        # We remove the subtensors from the list of parameters
        self.params = full_params

        logger.info("The cost gradient computation graph is built")

        self.step_rule = step_rule if step_rule else Scale()

        self.total_gradient_norm = named_copy(l2_norm(self.gradients.values()),
                                              "total_gradient_norm")
        self.steps, self.step_rule_updates = (
            self.step_rule.compute_steps(self.gradients))
        self.total_step_norm = named_copy(l2_norm(self.steps.values()),
                                          "total_step_norm")

    def initialize(self):
        all_updates = self.updates

        all_updates.extend([(param, param - self.steps[param]) for param in self.params if param not in self.subtensor_params])

        # Instead of substracting the gradient to the whole matrix, we only update the subtensor which is actually used
        for param, (subparam, canonized_indices, _, _) in self.subtensor_params.iteritems():
            new_value = tensor.inc_subtensor(param[canonized_indices], -self.steps[subparam])
            all_updates.append((param, new_value))

        all_updates.extend(self.step_rule_updates)
        self._function = theano.function(self.inputs, [], updates=all_updates)

class AdaDelta_SubtensorFix(AdaDelta):
    def __init__(self, subtensor_params={}, *args, **kwargs):
        super(AdaDelta_SubtensorFix, self).__init__(*args, **kwargs)
        self.subtensor_params = subtensor_params

    def compute_steps(self, previous_steps):
        subparams = [subparam for (subparam, _, _, _) in self.subtensor_params.values()]
        keys = [param for param in previous_steps if param not in subparams]
        parameter_wise = [self.compute_step(param, previous_steps[param]) for param in keys]
        
        # We use a special compute_step for lookup tables
        for param, (subparam, canonized_indices, _, _) in self.subtensor_params.iteritems():
            keys.append(subparam)
            parameter_wise.append(self.compute_step_subparam(param, canonized_indices, previous_steps[subparam]))

        steps, updates = equizip(*parameter_wise)
        steps = OrderedDict((param, step) for param, step 
                            in equizip(keys, steps))
        updates = list(itertools.chain(*updates))
        return steps, updates

    def compute_step_subparam(self, param, indices, previous_step):
        mean_square_step_tm1 = shared_floatx(param.get_value() * 0.)
        mean_square_delta_x_tm1 = shared_floatx(param.get_value() * 0.)
 
        # time is the number of step already computed (+1)
        time = theano.shared(numpy.int32(1))

        # last_updated contains the last time each row was updated
        last_updated = theano.shared(numpy.zeros(param.get_value().shape[0], dtype=numpy.int32))
        last_updated_sub = last_updated[indices]

        # We do the substraction as int on the cpu in order to mitigate some of the numeric instability
        lag = theano.sandbox.cuda.basic_ops.gpu_from_host(tensor.cast(time - last_updated_sub, dtype=theano.config.floatX))
        
        # We only update the relevant subtensors
        mean_square_delta_x_tm1_sub = mean_square_delta_x_tm1[indices]
        mean_square_step_tm1_sub = mean_square_step_tm1[indices]
 
        mean_square_step_t = tensor.set_subtensor(mean_square_step_tm1_sub,
                tensor.shape_padright(self.decay_rate ** lag) * mean_square_step_tm1_sub +
                (1 - self.decay_rate) * tensor.sqr(previous_step)
            )
 
        rms_delta_x_tm1 = tensor.sqrt(mean_square_delta_x_tm1_sub * tensor.shape_padright(self.decay_rate ** (lag-1.)) + self.epsilon)
        rms_step_t = tensor.sqrt(mean_square_step_t[indices] + self.epsilon)
        delta_x_t = rms_delta_x_tm1 / rms_step_t * previous_step
 
        mean_square_delta_x_t = tensor.set_subtensor(mean_square_delta_x_tm1_sub,
                tensor.shape_padright(self.decay_rate ** lag) * mean_square_delta_x_tm1_sub +
                (1 - self.decay_rate) * tensor.sqr(delta_x_t)
            )
 
        step = delta_x_t
        updates = [(mean_square_step_tm1, mean_square_step_t),
                   (mean_square_delta_x_tm1, mean_square_delta_x_t),
                   (last_updated, tensor.set_subtensor(last_updated_sub, time)),
                   (time, time+1)]
        return step, updates
