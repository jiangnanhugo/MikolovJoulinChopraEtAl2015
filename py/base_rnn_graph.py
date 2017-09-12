# Structurally Constrained Recurrent Network (SCRN) Model
#
# This gives an implementation of the SCRN model given in Mikolov et al. 2015, arXiv:1412.7753 [cs.NE], 
# https://arxiv.org/abs/1412.7753 using Python and Tensorflow.
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
from batch_generator import batch_generator
from log_prob import log_prob

# Define base RNN TensorFlow graph class
class base_rnn_graph(object):
    
    # Graph constructor
    def __init__(self, num_gpus, vocabulary_size, num_training_unfoldings, num_validation_unfoldings, training_batch_size,
                 validation_batch_size, optimization_frequency):
        
        # Input hyperparameters
        self._num_gpus = num_gpus
        self._num_training_unfoldings = num_training_unfoldings
        self._num_validation_unfoldings = num_validation_unfoldings
        self._optimization_frequency = optimization_frequency
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size
        self._vocabulary_size = vocabulary_size
        
        # Derived hyperparameters
        self._num_towers = self._num_gpus
        
        # Graph definition
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Setup tensor structures
            self._setup_cell_parameters()
            self._setup_training_data()
            self._setup_validation_data()
   
            # Optimizer hyperparameters
            self._clip_norm = tf.placeholder(tf.float32)
            self._learning_rate = tf.placeholder(tf.float32)
            self._momentum = tf.placeholder(tf.float32)
                
            # Optimizer
            self._optimizer = self._add_optimizer('momentum', self._learning_rate, self._momentum)
                    
            # Training:
            
            # Reset training state
            self._reset_training_state = self._reset_training_state_fun()
                
            # Train RNN on training data
            for i in range(self._num_training_unfoldings // self._optimization_frequency):
                training_labels = []
                training_outputs = []
                for tower in range(self._num_towers):
                    training_labels.append([])
                    training_outputs.append([])
                for tower in range(self._num_towers):
                    training_outputs[tower], training_labels[tower] = self._training_tower(i, tower, tower)
                all_training_outputs = []
                all_training_labels = []
                for tower in range(self._num_towers):
                    all_training_outputs += training_outputs[tower]
                    all_training_labels += training_labels[tower]
                logits = tf.concat(all_training_outputs, 0)
                labels = tf.concat(all_training_labels, 0)

                # Replace with hierarchical softmax in the future
                self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

                gradients, variables = zip(*self._optimizer.compute_gradients(self._cost))
                gradients, _ = tf.clip_by_global_norm(gradients, self._clip_norm)
                self._optimize = self._optimizer.apply_gradients(zip(gradients, variables))

            # Summarize training performance
            tf.summary.scalar('cost', self._cost)
            self._training_summary = tf.summary.merge_all()
                
            # Initialization:
            
            self._initialization = tf.global_variables_initializer()
                    
            # Validation:
    
            # Reset validation state
            self._reset_validation_state = self._reset_validation_state_fun()

            # Run RNN on validation data
            validation_outputs = []
            for tower in range(self._num_towers):
                validation_outputs.append([])
            for tower in range(self._num_towers):
                validation_outputs[tower] = self._validation_tower(tower,tower)
            logits = validation_outputs

            # Validation prediction, replace with hierarchical softmax in the future
            self._validation_prediction = tf.nn.softmax(logits)
        
    # Function to add choice of optimizer
    def _add_optimizer(self, optimizer, learning_rate, momentum):
        if optimizer == 'gradient_descent':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate, momentum)
        
    # Placeholder function for cell definition
    def _cell(self):
        print('Cell not defined')
        
    # Placeholder function to reset training state       
    def _reset_training_state_fun(self):
        print('Training state reset not defined')
    
    # Placeholder function to reset validation state
    def _reset_validation_state_fun(self):
        print('Validation state reset not defined')
    
    # Placeholder function to set up cell parameters
    def _setup_cell_parameters(self):
        print('Cell parameters not defined')  
                
    # Placeholder function to set up training parameters
    def _setup_training_parameters(self):
        print('Training parameters not defined')
        
    # Placeholder function to set up validation parameters
    def _setup_validation_parameters(self):
        print('Validation parameters not defined')
        
    # Placeholder function to implement a tower to run part of a batch of training data on a GPU
    def _training_tower(self, i, tower, gpu):
        print('Training tower not defined')
        
    # Placeholder function to implement a tower to run part of a batch of validation data on a GPU
    def _validation_tower(self, tower, gpu):
        print('Validation tower not defined')
            
    # Train model parameters
    def train(self, learning_rate, learning_decay, momentum, clip_norm, num_epochs, summary_frequency, training_text,
              validation_text,logdir):

        # Generate training batches
        print('Training Batch Generator:')
        training_batches = []
        for tower in range(self._num_towers):
            training_batches.append(batch_generator(tower, training_text[tower], self._training_batch_size,
                                                    self._num_training_unfoldings, self._vocabulary_size))
        
        # Generate validation batches
        print('Validation Batch Generator:')
        validation_batches = []
        tower = 0
        for tower in range(self._num_towers):
            validation_batches.append(batch_generator(tower, validation_text[tower], self._validation_batch_size,
                                                      self._num_validation_unfoldings, self._vocabulary_size))
        
        # Training loop
        batch_ctr = 0
        epoch_ctr = 0
        training_feed_dict = dict()
        validation_feed_dict = dict()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(graph=self._graph, config=config) as session:
            
            # Create summary writers
            training_writer = tf.summary.FileWriter(logdir + 'training/', graph=tf.get_default_graph())
            validation_writer = tf.summary.FileWriter(logdir + 'validation/', graph=tf.get_default_graph())
        
            # Initialize
            session.run(self._initialization)
            print('Initialized')

            # Iterate over fixed number of training epochs
            for epoch in range(num_epochs):

                # Display the learning rate for this epoch
                print('Epoch: %d  Learning Rate: %.2f' % (epoch+1, learning_rate))

                # Training Step:

                # Iterate over training batches
                for tower in range(self._num_towers):
                    training_batches[tower].reset_token_idx()
                session.run(self._reset_training_state)
                for batch in range(training_batches[0].num_batches()):

                    # Get next training batch
                    training_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        training_batches_next.append([])
                        training_batches_next[tower] = training_batches[tower].next()
                    batch_ctr += 1

                    # Optimization
                    training_feed_dict[self._clip_norm] = clip_norm
                    training_feed_dict[self._learning_rate] = learning_rate
                    training_feed_dict[self._momentum] = momentum
                    for tower in range(self._num_towers):
                        for i in range(self._num_training_unfoldings + 1):
                            training_feed_dict[self._training_data[tower][i]] = training_batches_next[tower][i]
                    _, summary = session.run([self._optimize, self._training_summary], feed_dict=training_feed_dict)

                    # Summarize current performance
                    training_writer.add_summary(summary, epoch * training_batches[0].num_batches() + batch)
                    
                    if (batch+1) % summary_frequency == 0:
                        cst = session.run(self._cost, feed_dict=training_feed_dict)
                        print('     Total Batches: %d  Current Batch: %d  Cost: %.2f' % 
                              (batch_ctr, batch+1, cst))
                      
                # Validation Step:
        
                # Iterate over validation batches
                for tower in range(self._num_towers):
                    validation_batches[tower].reset_token_idx()
                session.run(self._reset_validation_state)
                validation_log_prob_sum = 0
                for _ in range(validation_batches[0].num_batches()):
                    
                    # Get next validation batch
                    validation_batches_next = []
                    tower = 0
                    for tower in range(self._num_towers):
                        validation_batches_next.append([])
                        validation_batches_next[tower] = validation_batches[tower].next()
                    
                    # Validation
                    validation_batches_next_label = []
                    for tower in range(self._num_towers):
                        validation_batches_next_label_tmp = []
                        for i in range(self._num_validation_unfoldings):
                            validation_feed_dict[self._validation_input[tower][i]] = validation_batches_next[tower][i]
                            validation_batches_next_label_tmp.append(validation_batches_next[tower][i+1])
                        validation_batches_next_label.append(validation_batches_next_label_tmp)
                    validation_prediction = session.run(self._validation_prediction, feed_dict=validation_feed_dict)
                    
                    # Summarize current performance
                    for tower in range(self._num_towers):
                        for i in range(self._num_validation_unfoldings):
                            for j in range(self._validation_batch_size):
                                validation_log_prob_sum = validation_log_prob_sum + \
                                    log_prob(validation_prediction[tower][i][j], validation_batches_next_label[tower][i][j])
                    
                # Calculation validation perplexity
                N = self._num_towers * self._num_validation_unfoldings * \
                    validation_batches[0].num_batches() * self._validation_batch_size
                perplexity = float(2 ** (-validation_log_prob_sum / N))
                print('Epoch: %d  Validation Set Perplexity: %.2f' % (epoch+1, perplexity))

                # Update learning rate
                if epoch > 0 and perplexity > perplexity_last_epoch:
                    learning_rate *= learning_decay
                perplexity_last_epoch = perplexity