#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:10:22 2019

@author: dhh
"""

import numpy as np
import pandas as pd
#from layer_assist import Unit
#from Unit import call
import tensorflow as tf
dim = 20


def load_assist_data(dataset):
    sz_adj = pd.read_csv('%s_adj.csv'%dataset, header=None)
    adj = np.mat(sz_adj)
    data = pd.read_csv('sz_speed.csv')
    return data, adj

data, adj = load_assist_data('sz')
time_len = data.shape[0]
num_nodes = data.shape[1]

def preprocess_data(data1,  time_len, train_rate, seq_len, pre_len, model_name,scheme):
    train_size = int(time_len * train_rate)
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]            
            
    if model_name == 'tgcn':################TGCN###########################
        trainX, trainY, testX, testY = [], [], [], []
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            trainX.append(a1[0 : seq_len])
            trainY.append(a1[seq_len : seq_len + pre_len])
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            testX.append(b1[0 : seq_len])
            testY.append(b1[seq_len : seq_len + pre_len])         
            
    else:################AST-GCN###########################
        sz_poi = pd.read_csv('sz_poi.csv',header = None)
        sz_poi = np.transpose(sz_poi)
        sz_poi_max = np.max(np.max(sz_poi))
        sz_poi_nor = sz_poi/sz_poi_max
        sz_weather = pd.read_csv('sz_weather.csv',header = None)
        sz_weather = np.mat(sz_weather)
        sz_weather_max = np.max(np.max(sz_weather))
        sz_weather_nor = sz_weather/sz_weather_max
        sz_weather_nor_train = sz_weather_nor[0:train_size]
        sz_weather_nor_test = sz_weather_nor[train_size:time_len]

        if J == 1:#add poi(dim+1)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        if J == 2:#add weather(dim+11)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])
        else:#add kg(dim+12)
            trainX, trainY, testX, testY = [], [], [], []
            for i in range(len(train_data) - seq_len - pre_len):
                a1 = train_data[i: i + seq_len + pre_len]
                a2 = sz_weather_nor_train[i: i + seq_len + pre_len]
                a = np.row_stack((a1[0:seq_len],a2[0: seq_len + pre_len],sz_poi_nor[:1]))
                trainX.append(a)
                trainY.append(a1[seq_len : seq_len + pre_len])
            for i in range(len(test_data) - seq_len -pre_len):
                b1 = test_data[i: i + seq_len + pre_len]
                b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
                b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],sz_poi_nor[:1]))
                testX.append(b)
                testY.append(b1[seq_len : seq_len + pre_len])


    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)

    testY1 = np.array(testY)
    print(trainX1.shape)
    print(trainY1.shape)
    print(testX1.shape)
    print(testY1.shape)
    
    return trainX1, trainY1, testX1, testY1
    
class Unit():
    def __init__(self, dim, num_nodes, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b
#
#
#
class Unit1():
    def __init__(self, dim, num_nodes, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix=tf.convert_to_tensor(unit_matrix1)
        self.weight_unit1 ,self.bias_unit1 = self._emb(dim, time_len)
        
        x1=tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit1)
        x_output= tf.add(x1, self.bias_unit1)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit1 = tf.get_variable(name = 'weight_unit1', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit1 = tf.get_variable(name = 'bias_unit1', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w1 = weight_unit1
        bb = bias_unit1
        return w1, bb

class Unit2():
    def __init__(self, dim, num_nodes, time_len, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
        self.time_len = time_len
    def call(self, inputs, time_len):
        x, e = inputs
        x = np.transpose(x)
        x = x.astype(np.float64)
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        self.x_output = tf.add(x1, self.bias_unit)
#        print(x_output)
        self.x_output = tf.transpose(self.x_output)
#        x = x.astype(np.float64)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return self.x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            self.weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.time_len), dtype = tf.float32)
            self.bias_unit = tf.get_variable(name = 'bias_unit', shape =(num_nodes,1), initializer = tf.constant_initializer(dtype=tf.float32))
        self.w = self.weight_unit
        self.b = self.bias_unit
        return self.w, self.b


class Unit3():
    def __init__(self, dim, num_nodes, time_len, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
        self.time_len = time_len
    def call(self, inputs, time_len):
        x, e = inputs
        x = np.transpose(x)
        x = x.astype(np.float64)
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
        x_output = tf.transpose(x_output)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.time_len), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(num_nodes,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b

class Unit4():
    def __init__(self, dim, num_nodes, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix = tf.convert_to_tensor(unit_matrix1)
        self.weight_unit ,self.bias_unit = self._emb(dim, time_len)
        
        x1 = tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit)
        x_output = tf.add(x1, self.bias_unit)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            weight_unit = tf.get_variable(name = 'weight_unit', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            bias_unit = tf.get_variable(name = 'bias_unit', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        w = weight_unit
        b = bias_unit
        return w, b



class Unit5():
    def __init__(self, dim, num_nodes, reuse = None):
#        super(Unit, self).__init__(_reuse=reuse)
        self.dim = dim
        self.num_nodes = num_nodes
    def call(self, inputs, time_len):
        x, e = inputs  
        unit_matrix1 = tf.matmul(x, e)
        unit_matrix=tf.convert_to_tensor(unit_matrix1)
        self.weight_unit1 ,self.bias_unit1 = self._emb(dim, time_len)
        
        x1=tf.matmul(tf.cast(unit_matrix,tf.float32),self.weight_unit1)
        self.x_output= tf.add(x1, self.bias_unit1)
#        x_output = pd.DataFrame(x_output)
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes, time_len])
#        x_output = tf.reshape(x_output, shape=[-1, self.num_nodes * time_len])
        return self.x_output
    
    def _emb(self, dim, time_len):
        
        with tf.variable_scope('a',reuse = tf.AUTO_REUSE):
            self.weight_unit1 = tf.get_variable(name = 'weight_unit1', shape = (self.dim, self.num_nodes), dtype = tf.float32)
            self.bias_unit1 = tf.get_variable(name = 'bias_unit1', shape =(time_len,1), initializer = tf.constant_initializer(dtype=tf.float32))
        self.w1 = self.weight_unit1
        self.bb = self.bias_unit1
        return self.w1, self.bb


