# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the LSTM model for comparison with the SCRN model given in Mikolov et al. 2015,
# arXiv:1412.7753 [cs.NE], https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
#
# This model is superceded by the Delta-RNN model given in Ororbia et al. 2017, arXiv:1703.08864 [cs.CL], 
# https://arxiv.org/abs/1703.08864 implemented in this repository using Python and Tensorflow.
#
# This code fails to implement hierarchical softmax at this time as Tensorflow does not appear to include an
# implementation.  Hierarchical softmax can be included at a future date when hierarchical softmax is available 
# for Tensorflow.
#
# Stuart Hagler, 2017

# Imports
import math
import numpy as np
import tensorflow as tf

# Local imports
from base_rnn_graph import base_rnn_graph
from batch_generator import batch_generator
from log_prob import log_prob

# Define derived RNN TensorFlow graph class with saved hidden and output vectors
class base_rnn_graph2(base_rnn_graph):
    
    # Graph constructor
    def __init__(self, num_gpus, hidden_size, state_size, vocabulary_size, num_training_unfoldings,
                 num_validation_unfoldings, training_batch_size, validation_batch_size, optimization_frequency):
        
        # Input hyperparameters
        self._hidden_size = hidden_size
        self._state_size = state_size
        
        #
        base_rnn_graph.__init__(self, num_gpus, vocabulary_size, num_training_unfoldings, num_validation_unfoldings,
                                training_batch_size, validation_batch_size, optimization_frequency)
     
    #        
    def _reset_training_state_fun(self):
        return [ tf.group(self._training_output_saved[tower].assign(tf.zeros([self._training_batch_size, self._hidden_size])),
                           self._training_state_saved[tower].assign(tf.zeros([self._training_batch_size, 
                                                                              self._hidden_size]))) \
                  for tower in range(self._num_towers) ]
    
    #
    def _reset_validation_state_fun(self):
        return [ tf.group(self._validation_output_saved[tower].assign(tf.zeros([self._validation_batch_size,
                                                                                self._hidden_size])),
                           self._validation_state_saved[tower].assign(tf.zeros([self._validation_batch_size,
                                                                                self._hidden_size]))) \
                  for tower in range(self._num_towers) ]
            
    #
    def _setup_training_data(self):
        self._training_data = []
        self._training_output_saved = []
        self._training_state_saved = []
        for _ in range(self._num_towers):
            training_data_tmp = []
            for _ in range(self._num_training_unfoldings + 1):
                training_data_tmp.append(tf.placeholder(tf.float32, shape=[self._training_batch_size,
                                                                           self._vocabulary_size]))
            self._training_data.append(training_data_tmp)
            self._training_output_saved.append(tf.Variable(tf.zeros([self._training_batch_size, self._hidden_size]),
                                                           trainable=False))
            self._training_state_saved.append(tf.Variable(tf.zeros([self._training_batch_size, self._hidden_size]),
                                                          trainable=False))
            
    #
    def _setup_validation_data(self):
        self._validation_input = []
        self._validation_output_saved = []
        self._validation_state_saved = []
        for _ in range(self._num_towers):
            validation_input_tmp = []
            for _ in range(self._num_validation_unfoldings):
                validation_input_tmp.append(tf.placeholder(tf.float32, shape=[self._validation_batch_size,
                                                                              self._vocabulary_size]))
            self._validation_input.append(validation_input_tmp)
            self._validation_output_saved.append(tf.Variable(tf.zeros([self._validation_batch_size, self._hidden_size]),
                                                             trainable=False))
            self._validation_state_saved.append(tf.Variable(tf.zeros([self._validation_batch_size, self._hidden_size]),
                                                            trainable=False))
    
    # Implements a tower to run part of a batch of training data on a GPU
    def _training_tower(self, i, tower, gpu):
        
        with tf.device("/gpu:%d" % gpu):
        
            # Get saved training state
            output = self._training_output_saved[tower]
            state = self._training_state_saved[tower]

            # Run training data through cell
            labels = []
            outputs = []
            for j in range(self._optimization_frequency):
                x = self._training_data[tower][i*self._optimization_frequency + j]
                label = self._training_data[tower][i*self._optimization_frequency + j + 1]
                output, state = self._cell(x, output, state)
                labels.append(label)
                outputs.append(tf.nn.xw_plus_b(output, self._W, self._W_bias))

            # Save training state and return training outputs
            with tf.control_dependencies([self._training_output_saved[tower].assign(output), 
                                          self._training_state_saved[tower].assign(state)]):
                return outputs, labels
        
    # Implements a tower to run part of a batch of validation data on a GPU
    def _validation_tower(self, tower, gpu):
        
        with tf.device("/gpu:%d" % gpu):
        
            # Get saved validation state
            output = self._validation_output_saved[tower]
            state = self._validation_state_saved[tower]

            # Run validation data through cell
            outputs = []
            for i in range(self._num_validation_unfoldings):
                x = self._validation_input[tower][i]
                output, state = self._cell(x, output, state)
                outputs.append(tf.nn.xw_plus_b(output, self._W, self._W_bias))

            # Save validation state and return validation outputs
            with tf.control_dependencies([self._validation_output_saved[tower].assign(output), 
                                          self._validation_state_saved[tower].assign(state)]):
                return outputs