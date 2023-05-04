import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss
import os

def calculate_normalized_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2

    Args:
        adj: adj matrix

    Returns:
        np.ndarray: L
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_graph_for_message_passing(weight, transform_type='linear',dataset='METR_LA'):
    ricci_via_cost = load_graph_curvature(weight, dataset=dataset)
    if transform_type == 'linear':
        raise NotImplementedError
    else:
        ricci_via_cost = curvature_transform(ricci_via_cost, transform_type=transform_type, enhancing='negative') # negative, positive
        # ricci_via_cost = curvature_transform(ricci_via_cost, transform_type=transform_type, enhancing='positive') # negative, positive
    n = weight.shape[0]
    adj = weight + np.identity(n)

    # ricci_via_cost = sp.coo_matrix(ricci_via_cost)
    # adj = sp.coo_matrix(adj)


    d = np.sum(adj, axis=1)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = np.sqrt(np.linalg.inv(np.diag(d)))
    lap = np.matmul(np.matmul(sinvd, adj), sinvd)  # n*n
    lap_ricci = ricci_via_cost * lap
    # lap_ricci = np.expand_dims(lap_ricci, axis=0)
    return sp.coo_matrix(lap_ricci)
    # return lap_ricci

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

def random_adj_uniform(weight):
    n = weight.shape[0]
    print("Use Random Graph")
    return sp.coo_matrix(torch.rand(n,n))



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


class TCGNNCell(nn.Module):
    def __init__(self, num_units, adj_mx, num_nodes, device, dataset_name, input_dim=1):
        # ----------------------初始化参数---------------------------#
        super().__init__()
        self.num_units = num_units
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self._device = device
        self.act = torch.tanh

        # 这里提前构建好拉普拉斯
        support = calculate_graph_for_message_passing(adj_mx, transform_type='exp', dataset=dataset_name)
        # support = random_adj_uniform(adj_mx)

        # support = calculate_normalized_laplacian_with_ricci_curvature(adj_mx, transform_type='exp', dataset=dataset)
        self.normalized_adj = self._build_sparse_matrix(support, self._device)
        self.init_params()

    def init_params(self, bias_start=0.0):
        input_size = self.input_dim + self.num_units
        weight_0 = torch.nn.Parameter(torch.empty((input_size, 2 * self.num_units), device=self._device))
        bias_0 = torch.nn.Parameter(torch.empty(2 * self.num_units, device=self._device))
        weight_1 = torch.nn.Parameter(torch.empty((input_size, self.num_units), device=self._device))
        bias_1 = torch.nn.Parameter(torch.empty(self.num_units, device=self._device))

        torch.nn.init.xavier_normal_(weight_0)
        torch.nn.init.xavier_normal_(weight_1)
        torch.nn.init.constant_(bias_0, bias_start)
        torch.nn.init.constant_(bias_1, bias_start)

        self.register_parameter(name='weights_0', param=weight_0)
        self.register_parameter(name='weights_1', param=weight_1)
        self.register_parameter(name='bias_0', param=bias_0)
        self.register_parameter(name='bias_1', param=bias_1)

        self.weigts = {weight_0.shape: weight_0, weight_1.shape: weight_1}
        self.biases = {bias_0.shape: bias_0, bias_1.shape: bias_1}

    @staticmethod
    def _build_sparse_matrix(lap, device):
        lap = lap.tocoo()
        indices = np.column_stack((lap.row, lap.col))
        # this is to ensure row-major ordering to equal torch.sparse.sparse_reorder(L)
        indices = indices[np.lexsort((indices[:, 0], indices[:, 1]))]
        lap = torch.sparse_coo_tensor(indices.T, lap.data, lap.shape, device=device)
        return lap

    def forward(self, inputs, state):
        """
        Gated recurrent unit (GRU) with Graph Convolution.

        Args:
            inputs: shape (batch, self.num_nodes * self.dim)
            state: shape (batch, self.num_nodes * self.gru_units)

        Returns:
            torch.tensor: shape (B, num_nodes * gru_units)
        """
        output_size = 2 * self.num_units
        value = torch.sigmoid(
            self._gc(inputs, state, output_size, bias_start=1.0))  # (batch_size, self.num_nodes, output_size)
        r, u = torch.split(tensor=value, split_size_or_sections=self.num_units, dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self.num_units))  # (batch_size, self.num_nodes * self.gru_units)
        u = torch.reshape(u, (-1, self.num_nodes * self.num_units))

        c = self.act(self._gc(inputs, r * state, self.num_units))
        c = c.reshape(shape=(-1, self.num_nodes * self.num_units))
        new_state = u * state + (1.0 - u) * c
        return new_state

    def _gc(self, inputs, state, output_size, bias_start=0.0):
        """
        GCN

        Args:
            inputs: (batch, self.num_nodes * self.dim)
            state: (batch, self.num_nodes * self.gru_units)
            output_size:
            bias_start:

        Returns:
            torch.tensor: (B, num_nodes , output_size)
        """
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.dim)
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))  # (batch, self.num_nodes, self.gru_units)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, dim+gru_units, batch)
        x0 = x0.reshape(shape=(self.num_nodes, -1))

        x1 = torch.sparse.mm(self.normalized_adj.float(), x0.float())  # A * X

        x1 = x1.reshape(shape=(self.num_nodes, input_size, batch_size))
        x1 = x1.permute(2, 0, 1)  # (batch_size, self.num_nodes, input_size)
        x1 = x1.reshape(shape=(-1, input_size))  # (batch_size * self.num_nodes, input_size)

        weights = self.weigts[(input_size, output_size)]
        x1 = torch.matmul(x1, weights)  # (batch_size * self.num_nodes, output_size)

        biases = self.biases[(output_size,)]
        x1 += biases

        x1 = x1.reshape(shape=(batch_size, self.num_nodes, output_size))
        return x1


class TCGNN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        config['num_nodes'] = self.num_nodes
        self.input_dim = data_feature.get('feature_dim', 1)
        self.output_dim = data_feature.get('output_dim', 1)
        self.gru_units = int(config.get('rnn_units', 64))
        self.lam = config.get('lambda', 0.0015)


        self.dataset_name = config.get('dataset', 'METR_LA') #dataset name used to load or save ricci file

        super().__init__(config, data_feature)

        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')

        # -------------------构造模型-----------------------------
        self.tcgnn_model = TCGNNCell(self.gru_units, self.adj_mx, self.num_nodes, self.device, self.dataset_name, self.input_dim)
        self.output_model = nn.Linear(self.gru_units, self.output_window * self.output_dim)

    def forward(self, batch):
        """
        Args:
            batch: a batch of input,
                batch['X']: shape (batch_size, input_window, num_nodes, input_dim) \n
                batch['y']: shape (batch_size, output_window, num_nodes, output_dim) \n

        Returns:
            torch.tensor: (batch_size, self.output_window, self.num_nodes, self.output_dim)
        """
        inputs = batch['X']
        # labels = batch['y']

        batch_size, input_window, num_nodes, input_dim = inputs.shape
        inputs = inputs.permute(1, 0, 2, 3)  # (input_window, batch_size, num_nodes, input_dim)
        inputs = inputs.view(self.input_window, batch_size, num_nodes * input_dim).to(self.device)

        state = torch.zeros(batch_size, self.num_nodes * self.gru_units).to(self.device)
        for t in range(input_window):
            state = self.tcgnn_model(inputs[t], state)

        state = state.view(batch_size, self.num_nodes, self.gru_units)  # (batch_size, self.num_nodes, self.gru_units)
        output = self.output_model(state)  # (batch_size, self.num_nodes, self.output_window * self.output_dim)
        output = output.view(batch_size, self.num_nodes, self.output_window, self.output_dim)
        output = output.permute(0, 2, 1, 3)
        return output

    def calculate_loss(self, batch):
        # lam = self.lam
        # lreg = sum((torch.norm(param) ** 2 / 2) for param in self.parameters())

        labels = batch['y']
        y_predicted = self.predict(batch)

        y_true = self._scaler.inverse_transform(labels[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])

        # loss = torch.mean(torch.norm(y_true - y_predicted) ** 2 / 2) + lam * lreg
        # loss /= y_predicted.numel()
        # # return loss.masked_mae_torch(y_predicted, y_true, 0)
        # return loss
        # return loss.masked_mae_torch(y_predicted, y_true)
        return loss.masked_mse_torch(y_predicted, y_true)


    def predict(self, batch):
        return self.forward(batch)
