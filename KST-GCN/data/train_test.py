#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:09:29 2018

@author: dhh
"""

import numpy as np
from sklearn.cross_validation import train_test_split
import pandas as pd


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

 #首先，读取.CSV文件成矩阵的形式。
#my_matrix = np.loadtxt(open("shanghai_KG.txt"),delimiter=",",skiprows=0)
my_matrix=pd.read_csv(r'E:\csu\KG\TGCN\sz\sz_kg\sz_assist_kg.csv',names=['head','relation','tail'],encoding='GBK')
cols=list(my_matrix)
#print(cols)
cols.insert(1,cols.pop(cols.index('tail')))
#print(cols)
my_matrix=my_matrix.loc[:,cols]
#对于矩阵而言，将矩阵倒数第一列之前的数值给了X（输入数据），将矩阵大最后一列的数值给了y（标签）
#X, y = my_matrix[:,:-1],my_matrix[:,-1]
#利用train_test_split方法，将X,y随机划分问，训练集（X_train），训练集标签（X_test），测试卷（y_train）， 测试集标签（y_test），安训练集：测试集=7:3的概率划分，到此步骤，可以直接对数据进行处理
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train,test1 = train_test_split(my_matrix, test_size=0.02, random_state=42)
test,valid = train_test_split(test1, test_size=0.5, random_state=42)
 #此步骤，是为了将训练集与数据集的数据分别保存为CSV文件
 #np.column_stack将两个矩阵进行组合连接
#train= np.column_stack((X_train,y_train))
# #numpy.savetxt 将txt文件保存为。csv结尾的文件
#np.savetxt('train_usual.csv',train, delimiter = ',')
#test = np.column_stack((X_test, y_test))
#np.savetxt('test_usual.csv', test, delimiter = ',')
train.to_csv('train.txt',index=False,header=None,sep='\t')
test.to_csv('test.txt',index=False,header=None,sep='\t')
valid.to_csv('valid.txt',index=False,header=None,sep='\t')





