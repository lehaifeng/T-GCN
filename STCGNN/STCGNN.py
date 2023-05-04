import math
import os.path

import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
def calculate_scaled_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I

    Args:
        adj: adj_matrix

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)  # D
    lap = np.diag(d) - adj     # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    return 2 * lap / lam - np.eye(n)


def calculate_cheb_poly(lap, ks):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return np.asarray(lap_list[0:1])  # 1*n*n
    else:
        return np.asarray(lap_list)       # Ks*n*n


def calculate_first_approx(weight):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap = np.expand_dims(lap, axis=0)              # 1*n*n
    return lap

def load_graph_curvature(weight,dataset='PEMS_BAY'):
    ricci_via_cost = np.zeros((weight.shape[0], weight.shape[0]))
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

    ricci_file = os.path.join(PROJECT_ROOT, '../../../raw_data/{}/ricci_via_cost.npz'.format(dataset)) # 路径问题 {DATASET}/ricci_{DATASET}.npz

    print("DEBUG-cwd", os.path.exists(ricci_file) == True)
    print("Ricci file", ricci_file)
    if not os.path.exists(ricci_file):
        print("Not exist ricci")
        raise NotImplementedError("Not exist ricci")
    else:
        ricci_via_cost = np.load(ricci_file)
    return ricci_via_cost

def calculate_graph_for_message_passing(weight, transform_type='linear',dataset='METR_LA'):
    ricci_via_cost = load_graph_curvature(weight, dataset=dataset)
    if transform_type == 'linear':
        raise NotImplementedError
    else:
        ricci_via_cost = curvature_transform(ricci_via_cost, transform_type=transform_type, enhancing='negative') # negative, positive
        # ricci_via_cost = curvature_transform(ricci_via_cost, transform_type=transform_type, enhancing='positive') # negative, positive
    n = weight.shape[0]
    adj = weight + np.identity(n)
    d = np.sum(adj, axis=1)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap_ricci = ricci_via_cost * lap
    lap_ricci = np.expand_dims(lap_ricci, axis=0)
    return lap_ricci

def random_adj_uniform(weight):
    n = weight.shape[0]
    print("Use Random Graph")
    return np.expand_dims(torch.rand(n,n), axis=0)


def curvature_transform(adj_with_ricci, transform_type='linear', enhancing='positive'):
    """

    """
    print("Enhancing: {}".format(enhancing))
    if transform_type == 'linear':
        if enhancing == 'positive':
            min_ricci_weight = 0.0  # set the value of epsilon
            min_w_mul = abs(np.min(adj_with_ricci)) + min_ricci_weight
            adj_with_ricci = adj_with_ricci + min_w_mul

        else:
            adj_with_ricci = abs(np.max(adj_with_ricci)) - adj_with_ricci
        pass
    elif transform_type == 'exp':
        if enhancing == 'positive':
            adj_with_ricci = sigmoid(adj_with_ricci)
        else:
            adj_with_ricci = 1 - sigmoid(adj_with_ricci)
    np.fill_diagonal(adj_with_ricci, 1)
    return adj_with_ricci

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def load_ricci_curvature(file_name='raw_data/METR_LA/ricci_METR_LA.npz'):
    """
    TODO: LOAD ricci curvature from local disk
    """
    with open(file_name, 'rb') as f:
        adj_with_ricci = np.load(f)
    return adj_with_ricci
class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class ImprovedSpatioConvLayer(nn.Module):
    """
    嵌入ricci
    """
    def __init__(self, ks, c_in, c_out, lk, device):
        super(ImprovedSpatioConvLayer, self).__init__()
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, lk, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = ImprovedSpatioConvLayer(ks, c[1], c[1], lk, device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1)   # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)  # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc = FullyConvLayer(c, out_dim)

    def forward(self, x):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln)
        # (batch_size, input_dim(c), 1, num_nodes)
        return self.fc(x_t2)
        # (batch_size, output_dim, 1, num_nodes)


class STCGNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()

        self.dataset_name = config.get('dataset', 'METR_LA')
        self.Ks = config.get('Ks', 3)
        self.Kt = config.get('Kt', 3)
        self.blocks = config.get('blocks', [[1, 32, 64], [64, 32, 128]])
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.drop_prob = config.get('dropout', 0)
        self.lam = config.get('lambda', 0.0015)

        self.train_mode = config.get('stcgnn_train_mode', 'quick')  # or full
        if self.train_mode.lower() not in ['quick', 'full']:
            raise ValueError('STCGCN_train_mode must be `quick` or `full`.')
        self._logger.info('You select {} mode to train STCGNN model.'.format(self.train_mode))
        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')
        self.device = config.get('device', torch.device('cpu'))

        self.graph_conv_type = config.get('graph_conv_type', 'chebconv')
        self.transform_type = config.get('transform_type', 'linear')
        adj_mx = data_feature['adj_mx']  # ndarray
        # 计算GCN邻接矩阵的归一化拉普拉斯矩阵和对应的切比雪夫多项式或一阶近似
        if self.graph_conv_type.lower() == 'chebconv':
            laplacian_mx = calculate_scaled_laplacian(adj_mx)
            self.Lk = calculate_cheb_poly(laplacian_mx, self.Ks)
            self._logger.info('Chebyshev_polynomial_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
        elif self.graph_conv_type.lower() == 'gcnconv':
            self.Lk = calculate_first_approx(adj_mx)
            self._logger.info('First_approximation_Lk shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.Ks = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）
        elif self.graph_conv_type.lower() == 'rc_gcnconv':
            # self.Lk = calcuate_laplacian_normalized_by_ricci_curvature(adj_mx, transform_type=self.transform_type, dataset=self.dataset_name)
            self.Lk = calculate_graph_for_message_passing(adj_mx, transform_type=self.transform_type, dataset=self.dataset_name)

            self._logger.info('Enhanced laplacian shape: ' + str(self.Lk.shape))
            self._logger.info('Ricci transform type: '+ self.transform_type)
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.Ks = 1  # 一阶近似保留到K0和K1，但是不是数组形式，只有一个n*n矩阵，所以是1（本质上是2）
        elif self.graph_conv_type.lower() == 'random': # randomly
            self.Lk = random_adj_uniform(adj_mx)

            self._logger.info('Random Matrix shape: ' + str(self.Lk.shape))
            self.Lk = torch.FloatTensor(self.Lk).to(self.device)
            self.Ks = 1  # 一
        else:
            raise ValueError('Error graph_conv_type, must be chebconv or gcnconv.')

        # 模型结构
        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[0], self.drop_prob, self.Lk, self.device)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[1], self.drop_prob, self.Lk, self.device)
        self.output = OutputLayer(self.blocks[1][2], self.input_window - len(self.blocks) * 2
                                  * (self.Kt - 1), self.num_nodes, self.output_dim)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        x_st1 = self.st_conv1(x)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        outputs = self.output(x_st2)  # (batch_size, output_dim(1), output_length(1), num_nodes)
        outputs = outputs.permute(0, 2, 3, 1)  # (batch_size, output_length(1), num_nodes, output_dim)
        return outputs

    def calculate_loss(self, batch):
        lam = self.lam
        lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())
        if self.train_mode.lower() == 'quick':
            if self.training:  # 训练使用t+1时间步的loss
                y_true = batch['y'][:, 0:1, :, :]  # (batch_size, 1, num_nodes, feature_dim)
                y_predicted = self.forward(batch)  # (batch_size, 1, num_nodes, output_dim)
            else:  # 其他情况使用全部时间步的loss
                y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
                y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        else:   # 'full'
            y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
            y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        # return loss.masked_mse_torch(y_predicted, y_true, 0)
        return loss.masked_mse_torch(y_predicted, y_true)

        # return loss.masked_mae_torch(y_predicted, y_true, 0)
        # return loss.masked_mse_to
        # rch(y_predicted, y_true, 0) + lam * lreg

    def predict(self, batch):
        # 多步预测
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, num_nodes, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i+1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, num_nodes, output_dim)
        return y_preds
