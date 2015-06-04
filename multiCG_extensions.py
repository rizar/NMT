
import logging
import numpy
import os
import re

from toolz import merge

from blocks.algorithms import DifferentiableCostMinimizer
from blocks.dump import MainLoopDumpManager, save_parameter_values
from blocks.extensions import SimpleExtension
from blocks.monitoring.evaluators import AggregationBuffer
from blocks.extensions.monitoring import MonitoringExtension
from blocks.extensions.saveload import LoadFromDump, Dump

logger = logging.getLogger(__name__)


class IncrementalDump(SimpleExtension):

    def __init__(self, saveto, **kwargs):
        raise NotImplementedError("To be implemented!")
        super(IncrementalDump, self).__init__(**kwargs)
        self.saveto = saveto
        self.modelID = self._get_model_id(saveto)

    def _get_model_id(self, saveto):
        try:
            postfix = [int(m.group(1))
                       for m in [re.match(r'.*_([-0-9]+)', f)
                                 for f in os.listdir(saveto)]
                       if m is not None]
            model_id = max(postfix)
        except:
            model_id = 0
        return model_id

    def do(self, which_callback, *args):
        pass


class TrainingDataMonitoringWithMultiCG(SimpleExtension, MonitoringExtension):

    def __init__(self, variables, **kwargs):
        """Variables should be a list of list
        """
        num_cgs = len(variables)
        kwargs.setdefault("before_training", True)
        super(TrainingDataMonitoringWithMultiCG, self).__init__(**kwargs)
        self._buffers = []
        for i in xrange(num_cgs):
            self._buffers.append(
                AggregationBuffer(variables[i]
                                  if isinstance(variables[i], list)
                                  else [variables[i]],
                                  use_take_last=True))
        self._last_time_called = -1

    def do(self, callback_name, *args):
        if callback_name == 'before_training':
            for i in xrange(self.main_loop.num_cgs):
                if not isinstance(self.main_loop.algorithm.algorithms[i],
                                  DifferentiableCostMinimizer):
                    raise ValueError
                self.main_loop.algorithm.algorithms[i].add_updates(
                    self._buffers[i].accumulation_updates)
                self._buffers[i].initialize_aggregators()
        else:
            if (self.main_loop.status['iterations_done'] ==
                    self._last_time_called):
                raise Exception("TrainingDataMonitoring.do should be invoked"
                                " no more than once per iteration")
            self._last_time_called = self.main_loop.status['iterations_done']
            enc_id = numpy.argmax(args[0]['src_selector'])
            self.add_records(
                self.main_loop.log,
                self._buffers[enc_id].get_aggregated_values().items())
            self._buffers[enc_id].initialize_aggregators()


class SimpleTrainingDataMonitoringWithMultiCG(SimpleExtension,
                                              MonitoringExtension):

    def __init__(self, **kwargs):
        super(SimpleTrainingDataMonitoringWithMultiCG, self).__init__(**kwargs)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        if (self.main_loop.status['iterations_done'] ==
                self._last_time_called):
            raise Exception("TrainingDataMonitoring.do should be invoked"
                            " no more than once per iteration")
        self._last_time_called = self.main_loop.status['iterations_done']
        enc_id = numpy.argmax(args[0]['src_selector'])
        self.add_records(
            self.main_loop.log,
            self.main_loop.algorithm.retvals[enc_id].items())


class MainLoopDumpManagerWMT15(MainLoopDumpManager):

    def load_to(self, main_loop):
        """Loads the dump from the root folder into the main loop.

        Only difference from super().load_to is the exception handling
        for each step separately.
        """
        try:
            logger.info("Loading model parameters...")
            params_all = self.load_parameters()
            for i in xrange(main_loop.num_cgs):
                params_this = main_loop.models[i].get_params()
                missing = set(params_this) - set(params_all)
                for pname in params_this.keys():
                    if pname in params_all:
                        val = params_all[pname]
                        params_this[pname].set_value(val)
                        logger.info("Loaded to CG[{}] {:15}: {}"
                                    .format(i, val.shape, pname))
                    else:
                        logger.warning(
                            "Parameter does not exist: {}".format(pname))

                logger.info(
                    "Number of parameters loaded for computation graph[{}]: {}"
                    .format(i, len(params_this) - len(missing)))
        except Exception as e:
            logger.error("Error {0}".format(str(e)))

        try:
            logger.info("Loading iteration state...")
            main_loop.iteration_state = self.load_iteration_state()
        except Exception as e:
            logger.error("Error {0}".format(str(e)))

        try:
            logger.info("Loading log...")
            main_loop.log = self.load_log()
        except Exception as e:
            logger.error("Error {0}".format(str(e)))

    def dump_parameters(self, main_loop):
        params_to_save = []
        for i in xrange(main_loop.num_cgs):
            params_to_save.append(
                main_loop.models[i].get_param_values())
        save_parameter_values(merge(params_to_save),
                              self.path_to_parameters)


class DumpWithMultiCG(Dump):
    """Wrapper to use MainLoopDumpManagerWMT15"""
    def __init__(self, state_path, **kwargs):
        kwargs.setdefault("after_training", True)
        super(DumpWithMultiCG, self).__init__(state_path, **kwargs)
        self.manager = MainLoopDumpManagerWMT15(state_path)


class LoadFromDumpMultiCG(LoadFromDump):
    """Wrapper to use MainLoopDumpManagerWMT15"""

    def __init__(self, config_path, **kwargs):
        super(LoadFromDumpMultiCG, self).__init__(config_path, **kwargs)
        self.manager = MainLoopDumpManagerWMT15(config_path)
