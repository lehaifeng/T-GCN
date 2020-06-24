from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from utils import weight_variable_glorot,calculate_laplacian

flags = tf.app.flags
FLAGS = flags.FLAGS

class GCN(object):     
 
    def __init__(self, num_units, adj, inputs, output_dim, activation = tf.nn.tanh, 
                 input_size = None, num_proj=None, reuse = None, **kwargs):
        super(GCN, self).__init__(**kwargs)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._output_dim = output_dim
        self._inputs = inputs
        self._num_nodes = inputs.get_shape()[2].value
        self._input_dim = inputs.get_shape()[1].value ###seq_len
        self._batch_size = tf.shape(inputs)[0]
        self._adj = []  
        self._adj.append(calculate_laplacian(adj))
        self._activation = activation
        self._gconv()
        
    @staticmethod
    def _build_sparse_matrix(L):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        L = tf.SparseTensor(indices, L.data, L.shape)
        return tf.sparse_reorder(L)

    @property
    def output_size(self):
        output_size = self._num_units
        return output_size
        
    def init_state(self,batch_size):       
        state = tf.zeros(shape=[batch_size, self._num_nodes*self._num_units], dtype=tf.float32)
        return state  
               
    @staticmethod
    def _concat(x, x_):
        x_ = tf.expand_dims(x_, 0)
        return tf.concat([x, x_], axis=0)   
           
    def _gconv(self,scope=None):
        ####[batch, num_nodes, seq]
        inputs = self._inputs       
        inputs = tf.transpose(inputs, perm=[2,0,1])
#        print('0',inputs.get_shape())
        x0 = tf.reshape(inputs,shape=[self._num_nodes,self._batch_size*self._input_dim])
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            ####hidden1
            for adj in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(adj, x0)
            x1 = tf.reshape(x1,shape=[self._num_nodes,self._batch_size,self._input_dim])
            x1 = tf.reshape(x1,shape=[self._num_nodes*self._batch_size,self._input_dim])
            
            weights = weight_variable_glorot(self._input_dim, self.output_size, name='weights')
            self.hidden1 = self._activation(tf.matmul(x1, weights))
                      
            ####output            
            weights = weight_variable_glorot(self.output_size,self._output_dim, name='weights')
            self.output = tf.matmul(self.hidden1, weights)
            self.output = tf.reshape(self.output,shape=[self._num_nodes,self._batch_size,self._output_dim])
            self.output = tf.transpose(self.output, perm=[1,2,0])
            self.output = tf.reshape(self.output,shape=[-1,self._num_nodes])
#            print(self.output)
            
    
