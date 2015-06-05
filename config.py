
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
    config['plot'] = False
    config['bokeh_port'] = 5006

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
    config['dec_nhids'] = 1000
    config['enc_embed_0'] = 620
    config['enc_embed_1'] = 620
    config['dec_embed'] = 620
    config['representation_dim'] = 2000  # this is the joint annotation
                                         # dimension of encoders
    config['saveto'] = 'multiEnc_FIDE_multiCG_fast'

    # Optimization related
    config['batch_size_enc_0'] = 80
    config['batch_size_enc_1'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'Adam'
    config['learning_rate'] = 1e-4
    config['step_clipping'] = 5
    config['weight_scale'] = 0.01
    config['schedule'] = [24, 24]
    config['save_accumulators'] = True  # algorithms' update step variables
    config['load_accumulators'] = True  # be careful with this
    config['exclude_encs'] = [True, True]
    config['additional_excludes'] = \
        ['/decoder/sequencegeneratorwithmulticontext/readout/lookupfeedbackwmt15/lookuptable.W']

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0
    config['drop_input'] = [0.1, 0.1]

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fideen-en/'
    config['stream'] = 'fide-en'
    config['src_vocab_0'] = basedir + 'fi2en/vocab.fi.pkl'
    config['src_vocab_1'] = basedir + 'de2en/vocab.de.pkl'
    config['src_data_0'] = basedir + 'fi2en/all.tok.clean.seg.shuf2.fi-en.fi'
    config['src_data_1'] = basedir + 'de2en/all.tok.clean.shuf.split.de-en.de'
    config['src_vocab_size_0'] = 40001
    config['src_vocab_size_1'] = 200000

    config['trg_vocab'] = basedir + 'joint_vocab.sub.en.52k.pkl'
    config['trg_data_0'] = basedir + 'fi2en/all.tok.clean.shuf2.fi-en.en'
    config['trg_data_1'] = basedir + 'de2en/all.tok.clean.shuf.de-en.en'
    config['trg_vocab_size'] = 51546

    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    config['val_set_grndtruth_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    config['val_set_out_0'] = 'multiEnc_FIDE_multiCG/validation_out_0.txt'
    config['val_set_1'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.de'
    config['val_set_grndtruth_1'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.en'
    config['val_set_out_1'] = 'multiEnc_FIDE_multiCG/validation_out_1.txt'
    config['output_val_set'] = True
    config['beam_size'] = 12

    # Timing related
    config['reload'] = True
    config['save_freq'] = 5000
    config['sampling_freq'] = 13
    config['bleu_val_freq'] = 10000
    config['val_burn_in'] = 80000
    config['finish_after'] = 10000000

    # Monitoring related
    config['hook_samples'] = 2
    config['plot'] = False
    config['bokeh_port'] = 4444

    #return ReadOnlyDict(config)
    return config


def get_config_wmt15_fideen_en():
    config = {}

    # Model related
    config['num_encs'] = 3
    config['num_decs'] = 1
    config['seq_len'] = 50
    config['enc_nhids_0'] = 1000
    config['enc_nhids_1'] = 1000
    config['enc_nhids_2'] = 1000
    config['dec_nhids'] = 1000
    config['enc_embed_0'] = 620
    config['enc_embed_1'] = 620
    config['enc_embed_2'] = 620
    config['dec_embed'] = 620
    config['representation_dim'] = 2000  # this is the joint annotation
                                         # dimension of encoders
    config['saveto'] = 'multiEnc_FIDEEN_multiCG_fast'

    # Optimization related
    config['batch_size_enc_0'] = 80
    config['batch_size_enc_1'] = 80
    config['batch_size_enc_2'] = 80
    config['sort_k_batches'] = 12
    config['step_rule'] = 'Adam'
    config['learning_rate'] = 1e-4
    config['step_clipping'] = 5
    config['weight_scale'] = 0.01
    config['schedule'] = [24, 24, 12]
    config['save_accumulators'] = True  # algorithms' update step variables
    config['load_accumulators'] = True  # be careful with this
    config['exclude_encs'] = [True, True, False]
    config['additional_excludes'] = \
        ['/decoder/sequencegeneratorwithmulticontext/readout/lookupfeedbackwmt15/lookuptable.W']

    # Regularization related
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0
    config['drop_input'] = [0., 0., 0.1]

    # Vocabulary/dataset related
    basedir = '/data/lisatmp3/firatorh/nmt/wmt15/data/fideen-en/'
    config['stream'] = 'fideen-en'
    config['src_vocab_0'] = basedir + 'fi2en/vocab.fi.pkl'
    config['src_vocab_1'] = basedir + 'de2en/vocab.de.pkl'
    config['src_vocab_2'] = basedir + 'joint_vocab.sub.en.52k.pkl'
    config['src_data_0'] = basedir + 'fi2en/all.tok.clean.seg.shuf2.fi-en.fi'
    config['src_data_1'] = basedir + 'de2en/all.tok.clean.shuf.split.de-en.de'
    config['src_data_2'] = basedir + 'cs2en/all.tok.clean.shuf.cs-en.en'
    config['src_vocab_size_0'] = 40001
    config['src_vocab_size_1'] = 200000
    config['src_vocab_size_2'] = 40000

    config['trg_vocab'] = basedir + 'joint_vocab.sub.en.52k.pkl'
    config['trg_data_0'] = basedir + 'fi2en/all.tok.clean.shuf2.fi-en.en'
    config['trg_data_1'] = basedir + 'de2en/all.tok.clean.shuf.de-en.en'
    config['trg_data_2'] = basedir + 'cs2en/all.tok.clean.shuf.cs-en.en'
    config['trg_vocab_size'] = 51546

    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['bleu_script'] = '/data/lisatmp3/firatorh/turkishParallelCorpora/iwslt14/scripts/multi-bleu.perl'
    config['val_set_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.seg.fi'
    config['val_set_grndtruth_0'] = '/data/lisatmp3/firatorh/nmt/wmt15/data/fi-en/dev/newsdev2015.tok.en'
    config['val_set_out_0'] = 'multiEnc_FIDE_multiCG/validation_out_0.txt'
    config['val_set_1'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.de'
    config['val_set_grndtruth_1'] = '/data/lisatmp3/jeasebas/nmt/data/wmt15/full/dev/tok/newstest2013.tok.en'
    config['val_set_out_1'] = 'multiEnc_FIDE_multiCG/validation_out_1.txt'
    config['output_val_set'] = True
    config['beam_size'] = 12

    # Timing related
    config['reload'] = True
    config['save_freq'] = 5000
    config['sampling_freq'] = 13
    config['bleu_val_freq'] = 15000
    config['val_burn_in'] = 80000
    config['finish_after'] = 10000000

    # Monitoring related
    config['hook_samples'] = 2
    config['plot'] = False
    config['bokeh_port'] = 3333

    #return ReadOnlyDict(config)
    return config


def get_config_wmt15_fideen_en_TEST():

    config = get_config_wmt15_fideen_en()

    config['enc_nhids_0'] = 100
    config['enc_nhids_1'] = 100
    config['enc_nhids_2'] = 100
    config['dec_nhids'] = 100
    config['enc_embed_0'] = 62
    config['enc_embed_1'] = 62
    config['enc_embed_2'] = 62
    config['dec_embed'] = 62
    config['representation_dim'] = 200
    config['saveto'] = 'multiEnc_FIDEEN_multiCG_fast_TEST'

    config['src_vocab_size_0'] = 400
    config['src_vocab_size_1'] = 400
    config['src_vocab_size_2'] = 400
    config['trg_vocab_size'] = 515

    # Regularization related
    config['drop_input'] = [0.4, 0.7, 0.2]

    # Optimization related
    config['batch_size_enc_0'] = 8
    config['batch_size_enc_1'] = 8
    config['batch_size_enc_2'] = 8
    config['sort_k_batches'] = 12
    config['schedule'] = [1, 1, 1]

    # Timing related
    config['save_freq'] = 5
    config['finish_after'] = 10
    config['load_accumulators'] = False
    config['save_accumulators'] = False

    return config

