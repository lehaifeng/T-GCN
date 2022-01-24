# -*- coding: utf-8 -*-

#import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from utils import calculate_laplacian
import pandas as pd
import numpy as np

class tgcnCell(RNNCell):
    """Temporal Graph Convolutional Network """

    def call(self, inputs, **kwargs):
        pass

    def __init__(self, num_units, adj, num_nodes,dim, input_size=None,
                 act=tf.nn.tanh, reuse=None):

        super(tgcnCell, self).__init__(_reuse=reuse)
        self._act = act
        self._nodes = num_nodes
        self._units = num_units
        self._dim = dim
        self._adj = []
        self._adj.append(calculate_laplacian(adj))


    @property
    def state_size(self):
        return self._nodes * self._units

    @property
    def output_size(self):
        return self._units

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or "tgcn"):
            with tf.variable_scope("gates"):  
                value = tf.nn.sigmoid(
                    self._gc(inputs, state, 2 * self._units, bias=1.0, scope=scope))
                r, u = tf.split(value=value, num_or_size_splits=2, axis=1)
            with tf.variable_scope("candidate"):
                r_state = r * state
                c = self._act(self._gc(inputs, r_state, self._units, scope=scope))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


    def _gc(self, inputs, state, output_size, bias=0.0, scope=None):
        kgembedding = pd.read_csv('./data/sz_POI_KR-EAR(transR)_embedding5.csv',header = None)
#        sz_poi = np.transpose(kgembedding)
        kgembedding_max = np.max(np.max(kgembedding))
        kgembedding_nor = kgembedding/kgembedding_max
        
        x_unit = Unit_static(self._dim,self._nodes)
        embedding1 = kgembedding_nor.dropna(axis=1)
        embeddings = tf.convert_to_tensor(np.mat(embedding1 ,dtype=np.float32))
#        print('embeddings_shape:',embeddings.shape)
        inputs = x_unit.call([inputs, embeddings])
        ## inputs:(-1,num_nodes)
        inputs = tf.expand_dims(inputs, 2)
        ## state:(batch,num_node,gru_units)
        state = tf.reshape(state, (-1, self._nodes, self._units))
        ## concat
#        print('inputs_shape:',inputs.shape)
#        print('state_shape:',state.shape)
        x_s = tf.concat([inputs, state], axis=2)
#        print('x_s_shape:',x_s.shape)
        
        input_size = x_s.get_shape()[2].value
        ## (num_node,input_size,-1)
        x0 = tf.transpose(x_s, perm=[1, 2, 0])  
        x0 = tf.reshape(x0, shape=[self._nodes, -1])
        
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            for m in self._adj:
                x1 = tf.sparse_tensor_dense_matmul(m, x0)
#                print(x1)
            x = tf.reshape(x1, shape=[self._nodes, input_size,-1])
            x = tf.transpose(x,perm=[2,0,1])
            x = tf.reshape(x, shape=[-1, input_size])
            weights = tf.get_variable(
                'weights', [input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
            x = tf.matmul(x, weights)  # (batch_size * self._nodes, output_size)
            biases = tf.get_variable(
                "biases", [output_size], initializer=tf.constant_initializer(bias, dtype=tf.float32))
            x = tf.nn.bias_add(x, biases)
            x = tf.reshape(x, shape=[-1, self._nodes, output_size])
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
#            print('x_shape:',x.shape)
        return x
    
    
class Unit_static():
    def __init__(self, dim, num_nodes, act=tf.nn.relu, reuse = None):
        self.dim = dim
        self.num_nodes = num_nodes 
        self.act = act
    def call(self, inputs):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(self.dim)
        self.x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        self.x_output = tf.add(self.x1, self.bias_unit)
        return self.act(self.x_output)
    
    def _emb(self, dim):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            self.weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            self.bias_unit = tf.get_variable(name = 'bias_unit', shape =(self.num_nodes), initializer = tf.constant_initializer(dtype=tf.float32))
        self.w = self.weight_unit
        self.b = self.bias_unit
        return self.w, self.b
