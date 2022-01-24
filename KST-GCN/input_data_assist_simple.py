#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:10:22 2019

@author: dhh
"""

import numpy as np
import pandas as pd

def load_szassist_data(dataset):
    sz_adj = pd.read_csv(r'sz_data/sz_adj.csv',header=None)
    adj = np.mat(sz_adj)
    data = pd.read_csv(r'sz_data/sz_speed.csv')
    sz_tf1 = data.replace(0,np.nan) 
    sz_tf1 = sz_tf1.interpolate()
    sz_tf1 = sz_tf1.fillna(method='bfill')

    return sz_tf1, adj

def preprocess_data(data1, time_len, train_rate, seq_len, pre_len,methods,attribute):
    train_size = int(time_len * train_rate)        
    train_data = data1[0:train_size]
    test_data = data1[train_size:time_len]     
    trainX, trainY, testX, testY = [], [], [], []
    sz_poi = pd.read_csv(r'sz_data/sz_poi.csv',header = None)
    sz_poi = np.transpose(sz_poi)
    sz_poi_max = np.max(np.max(sz_poi))
    sz_poi_nor = sz_poi/sz_poi_max
    sz_weather = pd.read_csv(r'sz_data/sz_weather_all.csv',header = None)
    sz_weather = np.mat(sz_weather)
    sz_weather_max = np.max(np.max(sz_weather))
    sz_weather_nor = sz_weather/sz_weather_max
    sz_weather_nor_train = sz_weather_nor[0:train_size]
    sz_weather_nor_test = sz_weather_nor[train_size:time_len]
#    
    if methods == 'add kg':#add poi(dim+1)
        for i in range(len(train_data) - seq_len - pre_len):
            a1 = train_data[i: i + seq_len + pre_len]
            a2 = sz_weather_nor_train[i: i + seq_len+ pre_len]
            a = np.row_stack((a1[0:seq_len],a2[0: seq_len+ pre_len],sz_poi_nor[:1]))
            a = a.astype(np.float32)
            trainX.append(a)
            trainY.append(a1[seq_len : seq_len + pre_len])
        for i in range(len(test_data) - seq_len -pre_len):
            b1 = test_data[i: i + seq_len + pre_len]
            b2 = sz_weather_nor_test[i: i + seq_len + pre_len]
            b = np.row_stack((b1[0:seq_len],b2[0: seq_len + pre_len],sz_poi_nor[:1]))
            b = b.astype(np.float32)
            testX.append(b)
            testY.append(b1[seq_len : seq_len + pre_len])
#    elif attribute =='weather':
#        weather_embedding = pd.read_csv('sz_data/sz_weather_embedding.csv',header = None)
#        weather_embedding = abs(np.transpose(weather_embedding))
#        kg_embedding = pd.read_csv('sz_data/sz_kg_embedding.csv',header = None)
#        kg_embedding = abs(np.transpose(kg_embedding))
#        data1 = pd.DataFrame(data1).multiply(weather_embedding)
#        
#        for i in range(len(train_data) - seq_len - pre_len):
#            a1 = train_data[i: i + seq_len]
#            a1_assist = np.multiply(a1,weather_embedding)
#            a2 = train_data[i: i + seq_len + pre_len]
#            trainX.append(a1_assist[0 : seq_len])
#            trainY.append(a2[seq_len : seq_len + pre_len])
#        for i in range(len(test_data) - seq_len -pre_len):
#            b1 = test_data[i: i + seq_len]
#            b1_assist = np.multiply(b1,weather_embedding)
#            b2 = test_data[i: i + seq_len + pre_len]
#            testX.append(b1_assist[0 : seq_len])
#            testY.append(b2[seq_len : seq_len + pre_len])      
    else:
###############KTGCN###########################
        for i in range(len(train_data) - seq_len - pre_len+1):
            a = train_data[i: i + seq_len + pre_len]
            trainX.append(a[0 : seq_len])
            trainY.append(a[seq_len : seq_len + pre_len])
        for i in range(len(test_data) - seq_len -pre_len+1):
            b = test_data[i: i + seq_len + pre_len]
            testX.append(b[0 : seq_len])
            testY.append(b[seq_len : seq_len + pre_len])
        
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    
    return trainX1, trainY1, testX1, testY1
    
