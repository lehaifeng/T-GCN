# -*- coding: utf-8 -*-


import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_sz_data,load_los_data
from tgcn import tgcnCell
from gru import GRUCell 

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

time_start = time.time()

###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 1, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 100, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 7, 'time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 64, 'batch size.')
flags.DEFINE_string('dataset', 'sz', 'sz or los.')
flags.DEFINE_string('model_name', 'GRU', 'TGCN or GRU or GCN.')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate =  FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
if data_name == 'sz':
    data, adj = load_sz_data('sz')
if data_name == 'los':
    data, adj = load_los_data('los')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)
### Perturbation Analysis
#noise = np.random.normal(0,0.2,size=data.shape)
#noise = np.random.poisson(16,size=data.shape)
#scaler = MinMaxScaler()
#scaler.fit(noise)
#noise = scaler.transform(noise)
#data1 = data1 + noise
#### normalization
max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TGCN(_X, weights, biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)

    out = tf.concat(outputs, axis=0)
    out = tf.reshape(out, shape=[seq_len,-1,num_nodes,gru_units])
    out = tf.transpose(out, perm=[1,0,2,3])

    last_output,alpha = self_attention1(out, weight_att, bias_att)

    output = tf.reshape(last_output,shape=[-1,seq_len])
    output = tf.matmul(output, weights['out']) + biases['out']
    output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])

    return output, outputs, states, alpha
    
def self_attention1(x, weight_att,bias_att):
    x = tf.matmul(tf.reshape(x,[-1,gru_units]),weight_att['w1']) + bias_att['b1']
#    f = tf.layers.conv2d(x, ch // 8, kernel_size=1, kernel_initializer=tf.variance_scaling_initializer())
    f = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    g = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']
    h = tf.matmul(tf.reshape(x, [-1, num_nodes]), weight_att['w2']) + bias_att['b2']

    f1 = tf.reshape(f, [-1,seq_len])
    g1 = tf.reshape(g, [-1,seq_len])
    h1 = tf.reshape(h, [-1,seq_len])
    s = g1 * f1
    print('s',s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print('bata',beta)
    context = tf.expand_dims(beta,2) * tf.reshape(x,[-1,seq_len,num_nodes])

    context = tf.transpose(context,perm=[0,2,1])
    print('context', context)
    return context, beta 
def self_attention(x, ch, weight_att, bias_att):
    f = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w'])
    g = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w']) + bias_att['b_att']
    h = tf.matmul(tf.reshape(x, [-1, gru_units]), weight_att['w']) + bias_att['b_att']
    print('h',h)

    f = tf.reshape(f, [-1,num_nodes])
    g = tf.reshape(g, [-1,num_nodes])
    h = tf.reshape(h, [-1,num_nodes])    
    s = g * f
    print('s',s)

    beta = tf.nn.softmax(s, dim=-1)  # attention map
    print('bata',beta)
#    o = tf.matmul(beta, h) # [bs, N, C]
    o = beta * h
    print('o',o)
    gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

#    o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
#    o = tf.reshape(o, shape=x.shape)
    o = tf.expand_dims(o, 2)
    x = gamma * o + x
    print('x',x)
#    x = tf.reduce_sum(x, 2)
    return x, beta    
###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# weights
weights = {
    'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight_o')}
bias = {
    'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}
weight_att={
    'w1':tf.Variable(tf.random_normal([gru_units,1], stddev=0.1),name='att_w1'),
    'w2':tf.Variable(tf.random_normal([num_nodes,1], stddev=0.1),name='att_w2')}
bias_att = {
    'b1': tf.Variable(tf.random_normal([1]),name='att_b1'),
    'b2': tf.Variable(tf.random_normal([1]),name='att_b2')}

if model_name == 'TGCN_att':
    pred,ttto,ttts,alpha = TGCN(inputs, weights, bias)
#    print(alpha)
if model_name == 'GRU':
    pred,ttts,ttto = GRU(inputs, weights, bias)    
if model_name == 'GCN':
    model = GCN(gru_units, inputs, output_dim)
    
y_pred = pred
#print('ooooo',y_pred)
             

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

#out = 'out/%s'%(model_name)
out = 'out/%s'%(model_name)
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var

def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 
    return batch_s
    
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
  
for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output, alpha1 = sess.run([optimizer, loss, error, y_pred, alpha],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)

    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))
    
    if (epoch % 500 == 0):        
        saver.save(sess, path+'/model_100/graphGRU_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
#x = [i for i in range(training_epoch)]
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
#test_rmse = [float(i) for i in test_rmse]
var = pd.DataFrame(batch_loss1)
var.to_csv(path+'/batch_loss.csv',index = False,header = False)
var = pd.DataFrame(train_loss)
var.to_csv(path+'/train_loss.csv',index = False,header = False)
var = pd.DataFrame(batch_rmse1)
var.to_csv(path+'/batch_rmse.csv',index = False,header = False)
var = pd.DataFrame(train_rmse)
var.to_csv(path+'/train_rmse.csv',index = False,header = False)
var = pd.DataFrame(test_loss)
var.to_csv(path+'/test_loss.csv',index = False,header = False)
var = pd.DataFrame(test_acc)
var.to_csv(path+'/test_acc.csv',index = False,header = False)
var = pd.DataFrame(test_rmse)
var.to_csv(path+'/test_rmse.csv',index = False,header = False)


index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path+'/test_result.csv',index = False,header = False)
plot_result(test_result,test_label1,path)
plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

fig1 = plt.figure(figsize=(7,3))
ax1 = fig1.add_subplot(1,1,1)
plt.plot(np.sum(alpha1,0))
plt.savefig(path+'/alpha.jpg',dpi=500)
plt.show()


plt.imshow(np.mat(np.sum(alpha1,0)))
plt.savefig(path+'/alpha11.jpg',dpi=500)
plt.show()

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])
