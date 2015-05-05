
class ReadOnlyDict(dict):

    def __setitem__(self, key, value):
        raise(TypeError, "__setitem__ is not supported")

    def __delitem__(self, key):
        raise(TypeError, "__delitem__ is not supported")

    def update(self, d):
        raise(TypeError, "update is not supported")


def get_states_wmt15_fi_en_40k():
    state = {}

    # Model related
    state['seq_len'] = 50
    state['enc_nhids'] = 1000
    state['dec_nhids'] = 1000
    state['enc_embed'] = 620
    state['dec_embed'] = 620
    state['saveto'] = 'refBlocks3'

    # Optimization related
    state['batch_size'] = 80
    state['sort_k_batches'] = 12
    state['step_rule'] = 'AdaDelta'
    state['step_clipping'] = 10
    state['weight_scale'] = 0.01

    # Regularization related
    state['weight_noise_ff'] = False
    state['weight_noise_rec'] = False
    state['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    state['stream'] = 'fi-en'
    state['src_vocab'] = basedir + 'vocab.fi.pkl'
    state['trg_vocab'] = basedir + 'vocab.en.pkl'
    state['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    state['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    state['src_vocab_size'] = 40001
    state['trg_vocab_size'] = 40001
    state['unk_id'] = 1

    # Early stopping based on bleu related
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.seg.fi'
    state['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.en'
    state['val_set_out'] = 'refBlocks3/adadelta_40k_out.txt'
    state['output_val_set'] = True
    state['beam_size'] = 20

    # Timing related
    state['reload'] = True
    state['save_freq'] = 50
    state['sampling_freq'] = 1
    state['bleu_val_freq'] = 2000
    state['val_burn_in'] = 50000

    # Monitoring related
    state['hook_samples'] = 1

    #return ReadOnlyDict(state)
    return state


def get_states_wmt15_fi_en_TEST():
    state = {}

    # Model related
    state['seq_len'] = 50
    state['enc_nhids'] = 100
    state['dec_nhids'] = 100
    state['enc_embed'] = 62
    state['dec_embed'] = 62
    state['saveto'] = 'refBlocks3_TEST'

    # Optimization related
    state['batch_size'] = 8
    state['sort_k_batches'] = 12
    state['step_rule'] = 'AdaDelta'
    state['step_clipping'] = 10
    state['weight_scale'] = 0.01

    # Regularization related
    state['weight_noise_ff'] = False
    state['weight_noise_rec'] = False
    state['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    state['stream'] = 'fi-en'
    state['src_vocab'] = basedir + 'vocab.fi.pkl'
    state['trg_vocab'] = basedir + 'vocab.en.pkl'
    state['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    state['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    state['src_vocab_size'] = 501
    state['trg_vocab_size'] = 501
    state['unk_id'] = 1

    # Early stopping based on bleu related
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    state['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    state['val_set_out'] = 'refBlocks3_TEST/validation_out.txt'
    state['output_val_set'] = True
    state['beam_size'] = 2

    # Timing related
    state['reload'] = True
    state['save_freq'] = 1
    state['sampling_freq'] = 5
    state['bleu_val_freq'] = 10
    state['val_burn_in'] = 0

    # Monitoring related
    state['hook_samples'] = 1

    #return ReadOnlyDict(state)
    return state


def get_states_wmt15_fide_en_TEST():
    state = {}

    # Model related
    state['num_encs'] = 2
    state['seq_len'] = 50
    state['enc_nhids_0'] = 100
    state['enc_nhids_1'] = 100
    state['dec_nhids'] = 100
    state['enc_embed_0'] = 62
    state['enc_embed_1'] = 62
    state['dec_embed'] = 62
    state['saveto'] = 'multiEnc_TEST'

    # Optimization related
    state['batch_size_enc_0'] = 8
    state['batch_size_enc_1'] = 8
    state['sort_k_batches'] = 12
    state['step_rule'] = 'AdaDelta'
    state['step_clipping'] = 10
    state['weight_scale'] = 0.01
    state['schedule'] = [2, 3]

    # Regularization related
    state['weight_noise_ff'] = False
    state['weight_noise_rec'] = False
    state['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    state['stream'] = 'fide-en'
    state['src_vocab_0'] = basedir + 'vocab.fi.pkl'
    state['src_vocab_1'] = basedir + 'vocab.fi.pkl'
    state['src_data_0'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    state['src_data_1'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    state['src_vocab_size_0'] = 501
    state['src_vocab_size_1'] = 501

    state['trg_vocab'] = basedir + 'vocab.en.pkl'
    state['trg_data_0'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    state['trg_data_1'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    state['trg_vocab_size'] = 501

    state['unk_id'] = 1

    # Early stopping based on bleu related
    state['normalized_bleu'] = True
    state['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    state['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    state['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    state['val_set_out'] = 'multiEnc_TEST/validation_out.txt'
    state['output_val_set'] = True
    state['beam_size'] = 2

    # Timing related
    state['reload'] = True
    state['save_freq'] = 1
    state['sampling_freq'] = 50000
    state['bleu_val_freq'] = 10000
    state['val_burn_in'] = 0

    # Monitoring related
    state['hook_samples'] = 1

    #return ReadOnlyDict(state)
    return state
