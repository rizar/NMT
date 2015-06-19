
class ReadOnlyDict(dict):

    def __setitem__(self, key, value):
        raise(TypeError, "__setitem__ is not supported")

    def __delitem__(self, key):
        raise(TypeError, "__delitem__ is not supported")

    def update(self, d):
        raise(TypeError, "update is not supported")


def get_config_wmt15_fi_en_40k():
    config = {}

    # Model related
    config['seq_len'] = 50
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620
    config['saveto'] = 'refBlocks3'

    # Optimization related
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 10
    config['weight_scale'] = 0.01

    # Regularization related
    config['weight_noise_ff'] = 0.01
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'stream_fi_en'
    config['src_vocab'] = basedir + 'vocab.fi.pkl'
    config['trg_vocab'] = basedir + 'vocab.en.pkl'
    config['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    config['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    config['src_vocab_size'] = 40001
    config['trg_vocab_size'] = 40001
    config['unk_id'] = 1
    config['src_eos_idx'] = 40000
    config['trg_eos_idx'] = 40000

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.seg.fi'
    config['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_1.tok.en'
    config['val_set_out'] = 'refBlocks3/adadelta_40k_out.txt'
    config['output_val_set'] = True
    config['beam_size'] = 20

    # Timing related
    config['reload'] = True
    config['save_freq'] = 50
    config['sampling_freq'] = 1
    config['bleu_val_freq'] = 2000
    config['val_burn_in'] = 50000

    # Monitoring related
    config['hook_samples'] = 1

    #return ReadOnlyDict(config)
    return config


def get_config_wmt15_fi_en_TEST():
    config = {}

    # Model related
    config['seq_len'] = 50
    config['enc_nhids'] = 100
    config['dec_nhids'] = 100
    config['enc_embed'] = 62
    config['dec_embed'] = 62
    config['saveto'] = 'refBlocks3_TEST'

    # Optimization related
    config['batch_size'] = 8
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 10
    config['weight_scale'] = 0.01

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'stream_fi_en'
    config['src_vocab'] = basedir + 'vocab.fi.pkl'
    config['trg_vocab'] = basedir + 'vocab.en.pkl'
    config['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    config['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    config['src_vocab_size'] = 501
    config['trg_vocab_size'] = 501
    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    config['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    config['val_set_out'] = 'refBlocks3_TEST/validation_out.txt'
    config['output_val_set'] = True
    config['beam_size'] = 2

    # Timing related
    config['reload'] = True
    config['save_freq'] = 1
    config['sampling_freq'] = 5
    config['bleu_val_freq'] = 10
    config['val_burn_in'] = 0

    # Monitoring related
    config['hook_samples'] = 1

    #return ReadOnlyDict(config)
    return config


def get_config_wmt15_fi_en_40k_refMultiCG():
    config = {}

    # Model related
    config['seq_len'] = 50
    config['enc_nhids'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed'] = 620
    config['dec_embed'] = 620
    config['saveto'] = 'refMultiCG'

    # Optimization related
    config['batch_size'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 10
    config['weight_scale'] = 0.01

    # Regularization related
    config['weight_noise_ff'] = 0.01
    config['weight_noise_rec'] = False
    config['dropout'] = 0.5

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'stream_fi_en'
    config['src_vocab'] = basedir + 'vocab.fi.pkl'
    config['trg_vocab'] = basedir + 'vocab.en.pkl'
    config['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    config['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    config['src_vocab_size'] = 40001
    config['trg_vocab_size'] = 40001
    config['unk_id'] = 1
    config['src_eos_idx'] = 40000
    config['trg_eos_idx'] = 40000

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    config['val_set_grndtruth'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    config['val_set_out'] = config['saveto'] + '/adadelta_40k_out.txt'
    config['output_val_set'] = True
    config['beam_size'] = 20

    # Timing related
    config['reload'] = True
    config['save_freq'] = 1000
    config['sampling_freq'] = 13
    config['bleu_val_freq'] = 5000
    config['val_burn_in'] = 20000

    # Monitoring related
    config['hook_samples'] = 2

    return config
