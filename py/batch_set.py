# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# The batch generator class that is used to feed the data to the LSTM, SCRN, and SRN models.
#
# Stuart Hagler, 2017

# Imports
import numpy as np

#
class batch_set(object):
    
    #
    def __init__(self, num_towers, text, batch_size, num_unfoldings, vocabulary_size):
        
        #
        self._batch_size = batch_size
        self._num_towers = num_towers
        self._num_unfoldings = num_unfoldings
        self._text = text
        self._vocabulary_size = vocabulary_size
        
    #
        training_batches = []
        for tower in range(self._num_towers):
            training_batches.append(batch_generator(self._tower, training_text[self._tower], self._batch_size,
                                                    self._num_training_unfoldings, self._vocabulary_size))