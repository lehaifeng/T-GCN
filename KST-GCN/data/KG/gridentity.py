# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 16:49:26 2018

@author: Administrator
"""

import pandas as pd

gridentity2id=pd.read_csv(r'/DHH/DHH_KG/Mobike/MobikeKG/data/gridentity2id.txt',sep='\t',names=['grid','id'])
embedding=pd.read_csv(r'/DHH/DHH_KG/Mobike/MobikeKG/STransE/Datasets/data/STransE.s100.r0.0005.m5.l1_1.e500.entity2vec',sep='\t',header=None)
embedding=embedding.dropna(axis=1)
gridid=gridentity2id['id'].tolist()
grid_embedding=embedding.ix[gridid,:]

grid_embedding.to_csv('STransE_gridentity2vec.csv',index=False,header=None)
