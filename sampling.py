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


class BleuValidator(SimpleExtension):

    class ModelInfo:
        def __init__(self, bleu_score, path=None):
            self.bleu_score = bleu_score
            self.path = self._generate_path(path)

        def _generate_path(self, path):
            return '%s_best_bleu_model_%d_BLEU%.2f.npz' % \
                (path, int(time.time()), self.bleu_score) if path else None

    def __init__(self, source_sentence, samples, model, state,
                 data_stream, n_best=1, track_n_models=1, **kwargs):
        super(BleuValidator, self).__init__(**kwargs)
        self.source_sentence = source_sentence
        self.samples = samples
        self.model = model
        self.state = state
        self.data_stream = data_stream
        self.n_best = n_best
        self.track_n_models = track_n_models
        self.verbose = 'val_set_out' in state.keys()

        # Helpers
        self.vocab = data_stream.dataset.dictionary
        self.unk_sym = data_stream.dataset.unk_token
        self.eos_sym = data_stream.dataset.eos_token
        self.unk_idx = self.vocab[self.unk_sym]
        self.eos_idx = self.vocab[self.eos_sym]
        self.best_models = []
        self.val_bleu_curve = []
        self.beam_search = BeamSearch(beam_size=state['beam_size'],
                                      samples=samples)
        self.multibleu_cmd = ['perl', self.state['bleu_script'],
                              self.state['val_set_grndtruth'], '<']

        if self.state['reload']:
            try:
                bleu_score = numpy.load(self.state['prefix'] + 'val_bleu_scores.npz')
                self.val_bleu_curve = bleu_score['bleu_scores'].tolist()

                # Track n best previous bleu scores
                for bleu in sorted(self.val_bleu_curve)[-self.track_n_models]:
                    self.best_models.append(BleuValidator.ModelInfo(bleu))
                logger.debug("BleuScores Reloaded")
            except:
                logger.debug("BleuScores not Found")

    def do(self, which_callback, *args):

        # Track validation burn in
        if self.main_loop.status['iterations_done'] <= \
                self.state['val_burn_in']:
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
            ftrans = open(self.state['val_set_out'], 'w')

        for i, line in enumerate(self.data_stream.get_epoch_iterator()):
            """
            Load the sentence, retrieve the sample, write to file
            """

            seq = self._oov_to_unk(line[0])
            input_ = numpy.tile(seq, (self.state['beam_size'], 1))

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
        if min(self.best_bleus) < bleu_score:
            return True
        return False

    def _save_model(self, bleu_score):
        if self._is_valid_to_save(bleu_score):
            model = BleuValidator.ModelInfo(bleu_score, self.state['prefix'])

            # Manage n-best model list first
            if len(self.best_bleus) >= self.track_n_models:
                old_model = self.best_bleus[0]
                if old_model.path and os.path.isfile(old_model.path):
                    print "Deleting old model %s" % old_model.path
                    os.remove(old_model.path)
                self.best_bleus.remove(old_model)

            self.best_bleus.append(model)
            self.best_bleus.sort(key=operator.getattr('bleu_score'))

            # Save the model here
            s = signal.signal(signal.SIGINT, signal.SIG_IGN)
            print "Saving ...", model.path
            numpy.savez(model.path, **self.main_loop.model.get_param_values())
            numpy.savez(self.state['prefix'] + 'val_bleu_scores.npz',
                        bleu_scores=self.val_bleu_curve)
            signal.signal(signal.SIGINT, s)

    def _oov_to_unk(self, seq):
        return [x if x < self.state['src_vocab_size'] else self.unk_idx
                for x in seq]

    def _parse_input(self, line):
        seqin = line.split()
        seqlen = len(seqin)
        seq = numpy.zeros(seqlen+1, dtype='int64')
        for idx, sx in enumerate(seqin):
            seq[idx] = self.vocab.get(sx, self.unk_idx)
            if seq[idx] >= self.state['src_vocab_size']:
                seq[idx] = self.unk_idx
        seq[-1] = self.eos_idx
        return seq

