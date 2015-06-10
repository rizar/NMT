import logging
import numpy
import operator
import os
import re
import signal
import time
import theano

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch

from subprocess import Popen, PIPE
from toolz import merge

logger = logging.getLogger(__name__)


class SamplingBase(object):

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, eos_idx):
        try:
            return seq.tolist().index(eos_idx) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq):
        return [x if x < self.src_vocab_size else self.unk_idx
                for x in seq]

    def _parse_input(self, line, eos_idx):
        seqin = line.split()
        seqlen = len(seqin)
        seq = numpy.zeros(seqlen+1, dtype='int64')
        for idx, sx in enumerate(seqin):
            seq[idx] = self.vocab.get(sx, self.unk_idx)
            if seq[idx] >= self.src_vocab_size:
                seq[idx] = self.unk_idx
        seq[-1] = eos_idx
        return seq

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _isMultiCG(self, batch):
        if not hasattr(self.main_loop, 'num_cgs'):
            return False
        if not self.main_loop.num_cgs > 1:
            return False
        if not isinstance(self.enc_id, int):
            return False
        if not self.enc_id < self.main_loop.num_cgs:
            return False
        if 'src_selector' not in batch:
            return False
        if 'trg_selector' not in batch:
            return False
        return True


class Sampler(SimpleExtension, SamplingBase):
    """Samples from computation graph

        Does not use peeked batches
    """

    def __init__(self, model, data_stream, num_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, enc_id=None,
                 src_eos_idx=-1, trg_eos_idx=-1, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.data_stream = data_stream
        self.num_samples = num_samples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.enc_id = enc_id if enc_id is not None else ""
        self._synced = False
        self.sampling_fn = model.get_theano_function()

    def do(self, which_callback, *args):

        batch = args[0]

        # Get current model parameters
        if not self._synced:
            self._multiCG = self._isMultiCG(batch)
            if self._multiCG:
                sources = self._get_attr_rec(
                    self.main_loop.data_stream.streams[self.enc_id],
                    'data_stream')
                model_params = self.main_loop.models[self.enc_id].params
            else:
                sources = self._get_attr_rec(self.main_loop, 'data_stream')
                model_params = self.main_loop.model.params

            self.sources = sources
            self.model.params = model_params
            self._synced = True

        if self._multiCG:
            batch = self.main_loop.data_stream\
                .get_batch_with_stream_id(self.enc_id)

        batch_size = batch['source'].shape[0]

        # Get input names
        src_name = self.sources.sources[0]
        trg_name = self.sources.sources[1]

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = self.sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = self.sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
            self.src_ivocab[self.src_eos_idx] = '</S>'
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.trg_ivocab[self.trg_eos_idx] = '</S>'

        sample_idx = numpy.random.choice(
                batch_size, self.num_samples, replace=False)
        src_batch = batch[src_name]
        trg_batch = batch[trg_name]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Add inputs if selectors are specified
        # TODO: ordering matters here, fix it properly
        inputs = [input_]
        if self._multiCG:
            inputs += [batch['trg_selector']]
            inputs += [batch['src_selector']]

        # Sample
        _1, outputs, _2, _3, costs = (self.sampling_fn(*inputs))
        outputs = outputs.T
        costs = list(costs.T)

        print ""
        print "Sampling from computation graph[{}]".format(self.enc_id)
        for i in range(len(outputs)):
            input_length = self._get_true_length(input_[i], self.src_eos_idx)
            target_length = self._get_true_length(target_[i], self.trg_eos_idx)
            sample_length = self._get_true_length(outputs[i], self.trg_eos_idx)

            print "Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab)
            print "Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab)
            print "Sample: ", self._idx_to_word(outputs[i][:sample_length],
                                                self.trg_ivocab)
            print "Sample cost: ", costs[i][:sample_length].sum()
            print ""


class BleuValidator(SimpleExtension, SamplingBase):

    def __init__(self, samples, model, data_stream,
                 bleu_script, val_set_out, val_set_grndtruth, src_vocab_size,
                 src_selector=None, trg_selector=None,
                 n_best=1, track_n_models=1, trg_ivocab=None,
                 beam_size=5, val_burn_in=10000,
                 _reload=True, enc_id=None, saveto=None,
                 src_eos_idx=-1, trg_eos_idx=-1, **kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.samples = samples
        self.model = model
        self.data_stream = data_stream
        self.bleu_script = bleu_script
        self.val_set_out = val_set_out
        self.val_set_grndtruth = val_set_grndtruth
        self.src_vocab_size = src_vocab_size
        self.src_selector = src_selector
        self.trg_selector = trg_selector
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.trg_ivocab = trg_ivocab
        self.beam_size = beam_size
        self.val_burn_in = val_burn_in
        self._reload = _reload
        self.enc_id = enc_id if enc_id is not None else ""
        self.saveto = saveto if saveto else "."
        self.verbose = val_set_out
        self._synced = False
        self._multiCG = False

        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(beam_size=beam_size,
                                      samples=samples)
        self.multibleu_cmd = ['perl', bleu_script, val_set_grndtruth, '<']

        # Create saving directory if it does not exist
        if not os.path.exists(saveto):
            os.makedirs(saveto)

        if self._reload:
            try:
                bleu_score = numpy.load(
                        os.path.join(
                            saveto, 'val_bleu_scores{}.npz'.format(self.enc_id)))
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for i, bleu in enumerate(
                        sorted(self.val_bleu_curve, reverse=True)):
                    if i < self.track_n_models:
                        self.best_models.append(ModelInfo(bleu))
                logger.info("BleuScores Reloaded")
            except:
                logger.info("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.val_burn_in:
            return

        # Get current model parameters
        if not self._synced:
            batch = args[0]

            # Determine if using multiple Computation Graphs or not
            if hasattr(self.main_loop, 'num_cgs') and\
                    self.main_loop.num_cgs > 1 and\
                    'src_selector' in batch:
                if not isinstance(self.enc_id, int):
                    raise ValueError(
                        "Specify Computation Graph ID for BeamSearch")
                if not self.enc_id < self.main_loop.num_cgs:
                    raise ValueError(
                        "Invalid Computation Graph ID given in BeamSearch")
                if not self.src_selector:
                    raise ValueError(
                        "Source Selector variable not given in BeamSearch")
                if not self.trg_selector:
                    raise ValueError(
                        "Target Selector variable not given in BeamSearch")
                self._multiCG = True

            if self._multiCG:
                self.sources = self._get_attr_rec(
                    self.main_loop.data_stream.streams[self.enc_id],
                    'data_stream')
                self.model.params = self.main_loop.models[self.enc_id].params
            else:
                self.sources = self._get_attr_rec(
                    self.main_loop, 'data_stream')
                self.model.params = self.main_loop.model.params

            self._synced = True

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        logger.info("Started Validation: ")
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        # Get target vocabulary
        if not self.trg_ivocab:
            trg_vocab = self.sources.data_streams[1].dataset.dictionary
            self.trg_ivocab = {v: k for k, v in trg_vocab.items()}

        if self.verbose:
            ftrans = open(self.val_set_out, 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(line[0])
            inputs_dict = {self.source_sentence: numpy.tile(
                seq, (self.beam_size, 1))}

            # Branch for multiple computation graphs
            if self._multiCG:
                src_selector_input = numpy.zeros(
                    (self.main_loop.num_cgs,)).astype(theano.config.floatX)
                src_selector_input[self.enc_id] = 1.
                trg_selector_input = numpy.tile(
                    1., (1,)).astype(theano.config.floatX)
                inputs_dict.update(
                    {self.src_selector: src_selector_input,
                     self.trg_selector: trg_selector_input})

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(
                    input_values=inputs_dict,
                    max_length=3*len(seq), eol_symbol=self.trg_eos_idx,
                    ignore_first_eol=True)

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]

                    # convert idx to words
                    trans_out = self._idx_to_word(trans_out[:-1], self.trg_ivocab)

                except ValueError:
                    print "Can NOT find a translation for line: {}".format(i+1)
                    trans_out = '<UNK>'

                if j == 0:
                    # Write to subprocess and file if it exists
                    print >> mb_subprocess.stdin, trans_out
                    if self.verbose:
                        print >> ftrans, trans_out

            if i != 0 and i % 100 == 0:
                print "Translated {} lines of validation set...".format(i)

            mb_subprocess.stdin.flush()

        print "Total cost of the validation: {}".format(total_cost)
        self.data_stream.reset()
        if self.verbose:
            ftrans.close()

        # send end of file, read output.
        mb_subprocess.stdin.close()
        stdout = mb_subprocess.stdout.readline()
        print "output ", stdout
        out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
        logger.info("Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.))
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        print bleu_score
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
           key=operator.attrgetter('bleu_score')).bleu_score < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.saveto, self.enc_id)

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    logger.info("Deleting old model %s" % old_model.path)
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            logger.info("Saving new model {}".format(model.path))
            params_to_save = []
            if self._multiCG:
                for i in xrange(self.main_loop.num_cgs):
                    params_to_save.append(
                        self.main_loop.models[i].get_param_values())
                params_to_save = merge(params_to_save)
            else:
                params_to_save = self.main_loop.model.get_param_values()

            numpy.savez(model.path, **params_to_save)
            numpy.savez(
                os.path.join(
                    self.saveto,
                    'val_bleu_scores{}.npz'.format(self.enc_id)),
                bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)


class ModelInfo:
    def __init__(self, bleu_score, path=None, enc_id=None):
        self.bleu_score = bleu_score
        self.enc_id = enc_id if enc_id is not None else ''
        self.path = self._generate_path(path) if path else None

    def _generate_path(self, path):
        return os.path.join(
                path, 'best_bleu_model{}_{}_BLEU{:.2f}.npz'.format(
                    self.enc_id, int(time.time()), self.bleu_score))
