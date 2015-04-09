import numpy
import theano
import cPickle
import shelve

from blocks.graph import ComputationGraph
from blocks.extensions import TrainingExtension

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LargeVocabulary(TrainingExtension):

    def __init__(self, enc_embds, dec_embds,
                       out_embds, bias_embds,
                       enc_LV_len, dec_LV_len,
                       Dx_name, Dy_name, rolling_vocab_fname,
                       rng, mean, std,
                       src_name='english', trg_name='french'):
        """
        enc_embds : Theano shared variable
            Encoder embeddings
        dec_embds : Theano shared variable
            Decoder embeddings (feedback)
        out_embds : Theano shared variable
            Decoder embeddings (softmax)
        bias_embds : Theano shared variable
            Decoder biases (softmax)
        enc_LV_len : int
            Total number of source words
        dec_LV_len : int
            Total number of target words
        Dx_name : str
            Name of the source shelve
        Dy_name : str
            Name of the target shelve
        rolling_vocab_fname : str
            Name of the dictionary telling
            you when to change vocabularies
        rng : Numpy RandomState instance
        mean : float
        std : float
        src_name : str
            Default is 'english'
        trg_name : str
            Default is 'french'
        """
        super(LargeVocabulary, self).__init__()

        self.enc_embds = enc_embds
        self.dec_embds = dec_embds
        self.out_embds = out_embds
        self.bias_embds = bias_embds

        self.enc_LV_len = enc_LV_len
        self.dec_LV_len = dec_LV_len
        
        self.dim = enc_embds.get_value().shape[1]
        
        self.Dx = shelve.open(Dx_name)
        self.Dy = shelve.open(Dy_name)
        with open(rolling_vocab_fname, 'rb') as f:
            self.rolling_vocab_dict = cPickle.load(f)
        self.total_num_batches = max(self.rolling_vocab_dict)
        
        self.rng = rng
        self.mean = mean
        self.std = std
        
        self.src_name = src_name
        self.trg_name = trg_name

    def get_related(self, parameter):

        related = [parameter]
        tmp_cg = ComputationGraph(self.main_loop.algorithm.steps[parameter])
        for i,v in enumerate(tmp_cg.variables):
            if (v not in tmp_cg.parameters and
            isinstance(v, theano.tensor.sharedvar.TensorSharedVariable) and
            v.get_value().shape == parameter.get_value().shape):
                related.append(v)
        assert len(related) == 3
        return related

    def invert_dict(self, d):
        inv_d = {}
        for key in d:
            inv_d[d[key]] = key
        assert len(d) == len(inv_d) # Check for uniqueness
        return inv_d

    def roll_vocab_small2large(self):
        logger.info("Small to large")

        # Transfer from small to large parameters
        # self.enc_small, self.enc_large, etc. are lists
        
        for param, large_param in zip(self.enc_related, self.large_enc):
            large_param[self.enc_large] = param.get_value()[self.enc_small]

        for param, large_param in (zip(self.dec_related, self.large_dec) +
                                   zip(self.bias_related, self.large_bias)):
            large_param[self.dec_large] = param.get_value()[self.dec_small]

        for param, large_param in zip(self.out_related, self.large_out):
            large_param[:,self.dec_large] = param.get_value()[:,self.dec_small]

    def roll_vocab_update_dicts(self, new_large2small_src, new_large2small_trgt):
        logger.info("Update dicts")

        self.enc_l2s = new_large2small_src
        self.dec_l2s = new_large2small_trgt
        
        self.enc_s2l = self.invert_dict(self.enc_l2s)
        self.dec_s2l = self.invert_dict(self.dec_l2s)

        self.enc_large, self.enc_small = (list(elt) for elt in
                                          zip(*self.enc_l2s.items()))
        self.dec_large, self.dec_small = (list(elt) for elt in
                                          zip(*self.dec_l2s.items()))
    
    def roll_vocab_large2small(self):
        logger.info("Large to small")

        # Transfer from large to small parameters
        # self.enc_small, self.enc_large, etc. are lists
    
        for param, large_param in zip(self.enc_related, self.large_enc):
            tmp = numpy.empty((len(self.enc_l2s), self.dim), dtype=theano.config.floatX)
            tmp[self.enc_small] = large_param[self.enc_large]
            param.set_value(tmp)
            
        for param, large_param in (zip(self.dec_related, self.large_dec) +
                                   zip(self.bias_related, self.large_bias)):
            tmp = param.get_value()
            tmp[self.dec_small] = large_param[self.dec_large]
            param.set_value(tmp)

        for param, large_param in zip(self.out_related, self.large_out):
            tmp = param.get_value()
            tmp[:, self.dec_small] = large_param[:, self.dec_large]
            param.set_value(tmp)

    def replace_array_inplace(self, array, mapping):
        
        for i in xrange(numpy.shape(array)[0]): # Assumes array is 2-dimensional
            for j in xrange(numpy.shape(array)[1]):
                array[i,j] = mapping[array[i,j]]
                
    def before_training(self):
        logger.info("Initializing extra parameters")

        # Find out the Theano shared variables
        # that will have to be updated
        self.enc_related = self.get_related(self.enc_embds)
        self.dec_related = self.get_related(self.dec_embds)
        self.out_related = self.get_related(self.out_embds)
        self.bias_related = self.get_related(self.bias_embds)

        # Initialize large vocabulary parameters
        # These should be numpy ndarrays, not shared variables
        enc_embds_dim = self.enc_embds.get_value().shape[1]
        dec_embds_dim = self.dec_embds.get_value().shape[1]
        
        self.large_enc = 3*[numpy.zeros((self.enc_LV_len, enc_embds_dim),
                            dtype=theano.config.floatX)]
        self.large_dec = 3*[numpy.zeros((self.dec_LV_len, dec_embds_dim),
                            dtype=theano.config.floatX)]
        self.large_out = 3*[numpy.zeros((dec_embds_dim, self.dec_LV_len),
                            dtype=theano.config.floatX)]
        self.large_bias = 3*[numpy.zeros((self.dec_LV_len),
                             dtype=theano.config.floatX)]
        
        self.large_enc[0] = self.rng.normal(self.mean, self.std,
                            (self.enc_LV_len, enc_embds_dim))
        self.large_dec[0] = self.rng.normal(self.mean, self.std,
                            (self.dec_LV_len, dec_embds_dim))
        self.large_out[0] = self.rng.normal(self.mean, self.std,
                            (dec_embds_dim, self.dec_LV_len))

    def before_batch(self, batch):
        batch_id = self.main_loop.status['iterations_done'] % self.total_num_batches
        if batch_id in self.rolling_vocab_dict:
            if self.main_loop.status['iterations_done'] != 0:
                self.roll_vocab_small2large()
            self.roll_vocab_update_dicts(self.Dx[str(batch_id)],
                                         self.Dy[str(batch_id)])
            # May need to divide by number of batches
            self.roll_vocab_large2small()

        # Transform batch in-place
        
        self.replace_array_inplace(batch[self.src_name], self.enc_l2s)
        self.replace_array_inplace(batch[self.trg_name], self.dec_l2s)
