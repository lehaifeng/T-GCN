# -*- coding: utf-8 -*-

#import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.platform import tf_logging as logging


class GRUCell(RNNCell):
    """Gated Recurrent Units. """
    
    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, num_nodes, input_size=None,
                 act=tf.nn.tanh, reuse=None):
        """Gated Recurrent Units."""
        
        super(GRUCell, self).__init__(_reuse=reuse)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._act = act
        self._num_nodes = num_nodes
        self._num_units = num_units



    @property
    def state_size(self):
        return self._num_nodes * self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or "gru"):
            with tf.variable_scope("gates"): 
                # Reset gate and update gate.
                value = tf.nn.sigmoid(
                    self._linear(inputs, state, 2 * self._num_units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                c = self._linear(inputs, r * state, self._num_units, scope=scope)
#                c = self._linear(inputs, r * state, scope=scope)
                if self._act is not None:
                    c = self._act(c)
            new_h = u * state + (1 - u) * c
        return new_h, new_h
        
    
    def _linear(self, inputs, state, output_size, bias=0.0, scope=None):
        ## inputs:(-1,num_nodes)
#        print('1',inputs.get_shape())
        inputs = tf.expand_dims(inputs, 2)
#        print('2',inputs.get_shape())
       
        ## state:(batch,num_node,gru_units)
#        print('2',state.get_shape())
        state = tf.reshape(state, (-1, self._num_nodes, self._num_units))
#        print('2',state.get_shape())
        
        x_h  = tf.concat([inputs, state], axis=2)
        input_size = x_h.get_shape()[2].value
        
        x = tf.reshape(x_h, shape=[-1, input_size])

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias))

            x = tf.matmul(x, weights)  # (batch_size * self.num_nodes, output_size)          
            x = tf.nn.bias_add(x, biases)

            x = tf.reshape(x, shape=[-1, self._num_nodes ,output_size])
            x = tf.reshape(x, shape=[-1, self._num_nodes * output_size])
        return x  
