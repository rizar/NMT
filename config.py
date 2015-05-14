
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
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'fi-en'
    config['src_vocab'] = basedir + 'vocab.fi.pkl'
    config['trg_vocab'] = basedir + 'vocab.en.pkl'
    config['src_data'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi'
    config['trg_data'] = basedir + 'all.tok.clean.shuf.fi-en.en'
    config['src_vocab_size'] = 40001
    config['trg_vocab_size'] = 40001
    config['unk_id'] = 1

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
    config['stream'] = 'fi-en'
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


def get_config_wmt15_fide_en_TEST():
    config = {}

    # Model related
    config['num_encs'] = 2
    config['num_decs'] = 1
    config['seq_len'] = 50
    config['enc_nhids_0'] = 100
    config['enc_nhids_1'] = 150
    config['dec_nhids'] = 120
    config['enc_embed_0'] = 56
    config['enc_embed_1'] = 62
    config['dec_embed'] = 62
    config['src_rep_dim'] = 180  # Annotation dim for src_selector
    config['trg_rep_dim'] = 130  # Annotation dim for trg_selector
    config['representation_dim'] = 200  # this is the joint annotation
                                        # dimension of encoders
    config['saveto'] = 'multiEnc_TEST'

    # Optimization related
    config['batch_size_enc_0'] = 8
    config['batch_size_enc_1'] = 8
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 10
    config['weight_scale'] = 0.01
    config['schedule'] = [2, 3]

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/processed/'
    config['stream'] = 'fide-en'
    config['src_vocab_0'] = basedir + 'vocab.fi.pkl'
    config['src_vocab_1'] = basedir + 'vocab.fi.pkl'
    config['src_data_0'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi.TEST'
    config['src_data_1'] = basedir + 'all.tok.clean.shuf.seg1.fi-en.fi.TEST'
    config['src_vocab_size_0'] = 501
    config['src_vocab_size_1'] = 501

    config['trg_vocab'] = basedir + 'vocab.en.pkl'
    config['trg_data_0'] = basedir + 'all.tok.clean.shuf.fi-en.en.TEST'
    config['trg_data_1'] = basedir + 'all.tok.clean.shuf.fi-en.en.TEST'
    config['trg_vocab_size'] = 501

    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    config['val_set_1'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    config['val_set_grndtruth_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    config['val_set_grndtruth_1'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    config['val_set_out_0'] = 'multiEnc_TEST/validation_out_0.txt'
    config['val_set_out_1'] = 'multiEnc_TEST/validation_out_1.txt'
    config['output_val_set'] = True
    config['beam_size'] = 2

    # Timing related
    config['reload'] = True
    config['save_freq'] = 1
    config['sampling_freq'] = 100
    config['bleu_val_freq'] = 100
    config['val_burn_in'] = 0

    # Monitoring related
    config['hook_samples'] = 2

    #return ReadOnlyDict(config)
    return config


def get_config_wmt15_fide_en():
    config = {}

    # Model related
    config['num_encs'] = 2
    config['num_decs'] = 1
    config['seq_len'] = 50
    config['enc_nhids_0'] = 1000
    config['enc_nhids_1'] = 1000
    config['dec_nhids'] = 1200
    config['enc_embed_0'] = 620
    config['enc_embed_1'] = 620
    config['dec_embed'] = 620
    config['src_rep_dim'] = 500  # Annotation dim for src_selector
    config['trg_rep_dim'] = 200  # Annotation dim for trg_selector
    config['representation_dim'] = 2000  # this is the joint annotation
                                         # dimension of encoders
    config['saveto'] = 'multiEnc_FIDE_multiCG'

    # Optimization related
    config['batch_size_enc_0'] = 80
    config['batch_size_enc_1'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'AdaDelta'
    config['step_clipping'] = 5
    config['weight_scale'] = 0.01
    config['schedule'] = [24, 24]

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fide-en/'
    config['stream'] = 'fide-en'
    config['src_vocab_0'] = basedir + 'vocab.fi.pkl'
    config['src_vocab_1'] = basedir + 'vocab.de.pkl'
    config['src_data_0'] = basedir + 'all.tok.clean.seg.fi-en.fi'
    config['src_data_1'] = basedir + 'all.tok.clean.de-en.de'
    config['src_vocab_size_0'] = 40000
    config['src_vocab_size_1'] = 40000

    config['trg_vocab'] = basedir + 'joint_vocab.en.52k.pkl'
    config['trg_data_0'] = basedir + 'all.tok.clean.fi-en.en'
    config['trg_data_1'] = basedir + 'all.tok.clean.de-en.en'
    config['trg_vocab_size'] = 51896

    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    config['val_set_grndtruth_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    config['val_set_out_0'] = 'multiEnc_TEST/validation_out.txt'
    config['val_set_1'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.seg.fi'
    config['val_set_grndtruth_1'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015_TEST.tok.en'
    config['val_set_out_1'] = 'multiEnc_TEST/validation_out.txt'
    config['output_val_set'] = True
    config['beam_size'] = 2

    # Timing related
    config['reload'] = True
    config['save_freq'] = 100
    config['sampling_freq'] = 50000000
    config['bleu_val_freq'] = 10000000
    config['val_burn_in'] = 0

    # Monitoring related
    config['hook_samples'] = 1

    #return ReadOnlyDict(config)
    return config
