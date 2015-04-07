import logging
import numpy
import operator
import os
import re
import signal
import time

from blocks.extensions import SimpleExtension
from blocks.search import BeamSearch

from subprocess import Popen, PIPE

logger = logging.getLogger(__name__)


class Sampler(SimpleExtension):

    def __init__(self, model, data_stream, n_samples, batch_size,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.model = model
        self.data_stream = data_stream
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab

    def do(self, which_callback, *args):

        # Get current model parameters
        self.model.set_param_values(
            self.main_loop.model.get_param_values())

        # Get dictionaries, this may not be the practical way
        sources = self._get_attr_rec(self.main_loop, 'data_stream')

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}

        # Randomly select source samples from the current batch
        # WARNING: Source and target indices from data stream
        #  can be different
        batch = args[0]

        sample_idx = numpy.random.choice(self.batch_size,
                        self.n_samples, replace=False)
        src_batch = batch[self.main_loop.data_stream.mask_sources[0]]
        trg_batch = batch[self.main_loop.data_stream.mask_sources[1]]

        input_ = src_batch[sample_idx, :]
        target_ = trg_batch[sample_idx, :]

        # Sample
        _1, outputs, _2, _3, costs = (
            self.model.get_theano_function()(input_))
        outputs = outputs.T
        costs = list(costs.T)

        for i in range(len(outputs)):
            input_length = self._get_true_length(input_[i], self.src_vocab)
            target_length = self._get_true_length(target_[i], self.trg_vocab)
            sample_length = self._get_true_length(outputs[i], self.trg_vocab)

            print "Input : ", self._idx_to_word(input_[i][:input_length],
                                                self.src_ivocab)
            print "Target: ", self._idx_to_word(target_[i][:target_length],
                                                self.trg_ivocab)
            print "Sample: ", self._idx_to_word(outputs[i][:sample_length],
                                                self.trg_ivocab)
            print "Sample cost: ", costs[i][:sample_length].sum()
            print ""

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _get_true_length(self, seq, vocab):
        try:
            return seq.tolist().index(vocab['</S>']) + 1
        except ValueError:
            return len(seq)


class BleuValidator(SimpleExtension):

    def __init__(self, source_sentence, samples, model, beam_size, vocab_size,
                 data_stream, bleu_script, val_set_grndtruth, prefix,
                 val_burn_in=1000, val_set_out=None, reload_scores=False,
                 n_best=1, track_n_models=1, **kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.data_stream = data_stream
        self.bleu_script = bleu_script
        self.val_set_grndtruth = val_set_grndtruth
        self.prefix = prefix
        self.val_burn_in = val_burn_in
        self.val_set_out = val_set_out
        self.reload_scores = reload_scores
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.verbose = val_set_out

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(beam_size=self.beam_size,
                                      samples=samples)
        self.multibleu_cmd = ['perl', self.bleu_script,
                              self.val_set_grndtruth, '<']

        if self.reload_scores:
            try:
                bleu_score = numpy.load(self.prefix + 'val_bleu_scores.npz')
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for bleu in sorted(self.val_bleu_curve)[-self.track_n_models]:
                    self.best_models.append(ModelInfo(bleu))
                logger.debug("BleuScores Reloaded")
            except:
                logger.debug("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= self.val_burn_in:
            return

        # Get current model parameters
        self.model.set_param_values(
            self.main_loop.model.get_param_values())

        # Evaluate and save if necessary
        self._save_model(self._evaluate_model())

    def _evaluate_model(self):

        print "Started Validation: "
        val_start_time = time.time()
        mb_subprocess = Popen(self.multibleu_cmd, stdin=PIPE, stdout=PIPE)
        total_cost = 0.0

        if self.verbose:
            ftrans = open(self.val_set_out, 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(line[0])
            input_ = numpy.tile(seq, (self.beam_size, 1))

            # draw sample, checking to ensure we don't get an empty string back
            trans, costs = \
                self.beam_search.search(input_values={self.source_sentence:input_},
                    max_length=3*len(seq), eol_symbol=self.eos_idx,
                    ignore_first_eol=True)

            nbest_idx = numpy.argsort(costs)[:self.n_best]
            for j, best in enumerate(nbest_idx):
                try:
                    total_cost += costs[best]
                    trans_out = trans[best]
                except ValueError:
                    print "Could not fine a translation for line: {}".format(i+1)
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
        print "Validation Took: {} minutes".format(
            float(time.time() - val_start_time) / 60.)
        assert out_parse is not None

        # extract the score
        bleu_score = float(out_parse.group()[6:])
        self.val_bleu_curve.append(bleu_score)
        print bleu_score
        mb_subprocess.terminate()

        return bleu_score

    def _is_valid_to_save(self, bleu_score):
        if not self.best_models or min(self.best_models,
               key=operator.attrgetter('bleu_score')) < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = ModelInfo(bleu_score, self.prefix)

            # Manage n-best model list first
            if len(self.best_models) >= self.track_n_models:
                old_model = self.best_models[0]
                if old_model.path and os.path.isfile(old_model.path):
                    print "Deleting old model %s" % old_model.path
                    os.remove(old_model.path)
                self.best_models.remove(old_model)

            self.best_models.append(model)
            self.best_models.sort(key=operator.attrgetter('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            print "Saving ...", model.path
            numpy.savez(model.path, **self.main_loop.model.get_param_values())
            numpy.savez(self.prefix + 'val_bleu_scores.npz',
                        bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)

    def _oov_to_unk(self, seq):
        return [x if x < self.vocab_size else self.unk_idx
                for x in seq]

    def _parse_input(self, line):
        seqin = line.split()
        seqlen = len(seqin)
        seq = numpy.zeros(seqlen+1, dtype='int64')
        for idx, sx in enumerate(seqin):
            seq[idx] = self.vocab.get(sx, self.unk_idx)
            if seq[idx] >= self.vocab_size:
                seq[idx] = self.unk_idx
        seq[-1] = self.eos_idx
        return seq

class ModelInfo:
    def __init__(self, bleu_score, path=None):
        self.bleu_score = bleu_score
        self.path = self._generate_path(path)

    def _generate_path(self, path):
        return '%s_best_bleu_model_%d_BLEU%.2f.npz' % \
            (path, int(time.time()), self.bleu_score) if path else None
