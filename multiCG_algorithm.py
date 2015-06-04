
import logging
import numpy
import signal
import theano
import traceback

from collections import OrderedDict
from theano import tensor

from blocks import config as cfg
from blocks.algorithms import (GradientDescent, StepClipping,
                               CompositeRule, variable_mismatch_error,
                               DifferentiableCostMinimizer,
                               StepRule)
from blocks.main_loop import (MainLoop, TrainingFinish,
                              error_in_error_handling_message,
                              error_message)
from blocks.theano_expressions import l2_norm
from blocks.utils import (change_recursion_limit, reraise_as, shared_floatx)
from blocks.utils.profile import Timer

logger = logging.getLogger(__name__)


class StepClippingWithRemoveNotFinite(StepRule):

    def __init__(self, threshold=None, scale=0.1):
        if threshold:
            self.threshold = shared_floatx(threshold)
        self.scale = scale

    def compute_steps(self, previous_steps):
        if not hasattr(self, 'threshold'):
            return previous_steps
        norm = l2_norm(previous_steps.values())
        multiplier = tensor.switch(tensor.ge(norm, self.threshold),
                                   self.threshold / norm, 1)
        notfinite = tensor.or_(tensor.isnan(norm), tensor.isinf(norm))
        steps = OrderedDict(
            (param, tensor.switch(
                notfinite, param * self.scale, step * multiplier))
            for param, step in previous_steps.items())
        return steps, []


class MainLoopWithMultiCG(MainLoop):

    def __init__(self, models, **kwargs):
        self.models = models
        self.num_cgs = len(models)
        # TODO: fix this
        kwargs['model'] = models[0]
        super(MainLoopWithMultiCG, self).__init__(**kwargs)

    def run(self):
        logging.basicConfig()

        for i in xrange(self.num_cgs):
            if self.models[i] and isinstance(self.algorithm.algorithms[i],
                                             DifferentiableCostMinimizer):
                if not self.models[i].get_objective() ==\
                        self.algorithm.algorithms[i].cost:
                    logger.warning(
                        "different costs for model {} and algorithm {}"
                        .format(i, i))
                if not (set(self.models[i].get_params().values()) ==
                        set(self.algorithm.algorithms[i].params)):
                    logger.warning(
                        "different params for model {} and algorithm {}"
                        .format(i, i))

        with change_recursion_limit(cfg.recursion_limit):
            self.original_sigint_handler = signal.signal(
                signal.SIGINT, self._handle_epoch_interrupt)
            self.original_sigterm_handler = signal.signal(
                signal.SIGTERM, self._handle_batch_interrupt)
            try:
                logger.info("Entered the main loop")
                if not self.status['training_started']:
                    for extension in self.extensions:
                        extension.main_loop = self
                    self._run_extensions('before_training')
                    with Timer('initialization', self.profile):
                        self.algorithm.initialize()
                    self.status['training_started'] = True
                if self.log.status['iterations_done'] > 0:
                    self._run_extensions('on_resumption')
                    self.status['epoch_interrupt_received'] = False
                    self.status['batch_interrupt_received'] = False
                with Timer('training', self.profile):
                    while self._run_epoch():
                        pass
            except TrainingFinish:
                self.log.current_row['training_finished'] = True
            except Exception as e:
                self._restore_signal_handlers()
                self.log.current_row['got_exception'] = traceback.format_exc(e)
                logger.error("Error occured during training." + error_message)
                try:
                    self._run_extensions('on_error')
                except Exception as inner_e:
                    logger.error(traceback.format_exc(inner_e))
                    logger.error("Error occured when running extensions." +
                                 error_in_error_handling_message)
                reraise_as(e)
            finally:
                if self.log.current_row.get('training_finished', False):
                    self._run_extensions('after_training')
                if cfg.profile:
                    self.profile.report()
                self._restore_signal_handlers()


class GradientDescentWithMultiCG(object):

    def __init__(self, costs, params, step_rule, **kwargs):
        self.num_cgs = len(costs)
        self.algorithms = []
        self._functions = []

        for i in xrange(len(costs)):
            self.algorithms.append(
                GradientDescent(
                    cost=costs[i], params=params[i],
                    step_rule=step_rule))

    def initialize(self):

        # Check if both computation graphs have identical inputs
        inputs = set.intersection(
            *[set(self.algorithms[i].inputs)
                for i in xrange(self.num_cgs)])
        if not all(
                [set(self.algorithms[i].inputs) == inputs
                    for i in xrange(self.num_cgs)]):
            raise ValueError(
                "mismatch of input names between computation graphs")

        for i in xrange(self.num_cgs):
            logger.info("Initializing the training algorithm {}".format(i))
            all_updates = self.algorithms[i].updates
            for param in self.algorithms[i].params:
                all_updates.append(
                    (param, param - self.algorithms[i].steps[param]))
            all_updates += self.algorithms[i].step_rule_updates
            self._functions.append(theano.function(
                self.algorithms[i].inputs, [], updates=all_updates))
            logger.info("The training algorithm {} is initialized".format(i))

    def process_batch(self, batch):
        for i in xrange(self.num_cgs):
            if not set(batch.keys()) == set(
                    [v.name for v in self.algorithms[i].inputs]):
                raise ValueError(
                    "mismatch of variable names and data sources" +
                    variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=[v.name for v in
                                   self.algorithms[i].inputs]))
        cg_id = numpy.argmax(batch['src_selector'])
        ordered_batch = [batch[v.name] for v in self.algorithms[cg_id].inputs]
        self._functions[cg_id](*ordered_batch)


class GradientDescentWithMultiCGandMonitors(object):

    def __init__(self, costs, params, step_rule, **kwargs):
        self.num_cgs = len(costs)
        self.algorithms = []
        self._functions = []
        self.retvals = [dict() for _ in xrange(self.num_cgs)]

        for i in xrange(len(costs)):
            self.algorithms.append(
                GradientDescent(
                    cost=costs[i], params=params[i],
                    step_rule=step_rule))

    def initialize(self):

        # Check if both computation graphs have identical inputs
        inputs = set.intersection(
            *[set(self.algorithms[i].inputs)
                for i in xrange(self.num_cgs)])
        if not all(
                [set(self.algorithms[i].inputs) == inputs
                    for i in xrange(self.num_cgs)]):
            raise ValueError(
                "mismatch of input names between computation graphs")

        for i in xrange(self.num_cgs):
            logger.info("Initializing the training algorithm {}".format(i))
            all_updates = self.algorithms[i].updates
            for param in self.algorithms[i].params:
                all_updates.append(
                    (param, param - self.algorithms[i].steps[param]))
            all_updates += self.algorithms[i].step_rule_updates
            self._functions.append(theano.function(
                self.algorithms[i].inputs,
                [self.algorithms[i].cost,
                 self.algorithms[i].total_gradient_norm,
                 self.algorithms[i].total_step_norm], updates=all_updates))
            logger.info("The training algorithm {} is initialized".format(i))

    def process_batch(self, batch):
        for i in xrange(self.num_cgs):
            if not set(batch.keys()) == set(
                    [v.name for v in self.algorithms[i].inputs]):
                raise ValueError(
                    "mismatch of variable names and data sources" +
                    variable_mismatch_error.format(
                        sources=batch.keys(),
                        variables=[v.name for v in
                                   self.algorithms[i].inputs]))
        cg_id = numpy.argmax(batch['src_selector'])
        ordered_batch = [batch[v.name] for v in self.algorithms[cg_id].inputs]
        rvals = self._functions[cg_id](*ordered_batch)
        self.retvals[cg_id]['cost'] = float(rvals[0])
        self.retvals[cg_id]['total_gradient_norm'] = float(rvals[1])
        self.retvals[cg_id]['total_step_norm'] = float(rvals[2])
