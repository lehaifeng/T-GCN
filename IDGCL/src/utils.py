from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikiCS
import torch.nn.functional as F
import os.path as osp
import os
import numpy as np
import torch
import copy
import torch.nn as nn
from datetime import datetime


def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','root','epochs','isAnneal','dropout','warmup_step','clus_num_iters']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def decide_config(root, dataset):
    """
    Create a configuration to download datasets
    :param root: A path to a root directory where data will be stored
    :param dataset: The name of the dataset to be downloaded
    :return: A modified root dir, the name of the dataset class, and parameters associated to the class
    """
    dataset = dataset.lower()
    if dataset == 'cora' or dataset == 'citeseer' or dataset == "pubmed":
        # root = osp.join(root, "pyg", "planetoid")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Planetoid, "src": "pyg"}
    elif dataset == "computers":
        dataset = "Computers"
        # root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "photo":
        dataset = "Photo"
        # root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Amazon, "src": "pyg"}
    elif dataset == "cs" :
        dataset = "CS"
        # root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "physics":
        dataset = "Physics"
        # root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root, "name": dataset},
                  "name": dataset, "class": Coauthor, "src": "pyg"}
    elif dataset == "wikics":
        dataset = "WikiCS"
        # root = osp.join(root, "pyg")
        params = {"kwargs": {"root": root},
                  "name": dataset, "class": WikiCS, "src": "pyg"}
    else:
        raise Exception(
            f"Unknown dataset name {dataset}, name has to be one of the following 'cora', 'citeseer', 'pubmed', 'photo', 'computers', 'cs', 'physics'")
    return params


def create_masks(data, name=None):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    if not hasattr(data, "val_mask"):

        data.train_mask = data.dev_mask = data.test_mask = None  # 1: 1: 8

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * 0.1)
            test_size = int(labels.shape[0] * 0.8)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if not hasattr(data, "train_mask"):
                data.train_mask = train_mask
                data.val_mask = dev_mask
                data.test_mask = test_mask
            else:
                data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
                data.val_mask = torch.cat((data.val_mask, dev_mask), dim=0)
                data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)
    elif name == 'cora':
        train_mask, val_mask, test_mask = data.train_mask.reshape(1, -1), \
                                          data.val_mask.reshape(1, -1), \
                                          data.test_mask.reshape(1, -1)
        data.train_mask = data.train_mask.reshape(1, -1)
        data.val_mask = data.val_mask.reshape(1, -1)
        data.test_mask = data.test_mask.reshape(1, -1)
        for i in range(19):
            data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
            data.val_mask = torch.cat((data.val_mask, val_mask), dim=0)
            data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    else:  # in the case of WikiCS
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T

    return data


def create_specific_label(data, train_ratio):
    data.train_mask = data.val_mask = data.test_mask = None

    for i in range(20):
        labels = data.y.numpy()
        train_size = int(labels.shape[0] * train_ratio)

        perm = np.random.permutation(labels.shape[0])
        trian_index = perm[:train_size]
        test_index = perm[train_size:]

        data_index = np.arange(labels.shape[0])
        train_mask = torch.tensor(np.in1d(data_index, trian_index), dtype=torch.bool).reshape(1, -1)
        test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool).reshape(1, -1)

        if not hasattr(data, "train_mask"):
            data.train_mask = train_mask
            data.test_mask = test_mask
        else:
            data.train_mask = torch.cat((data.train_mask, train_mask), dim=0)
            data.test_mask = torch.cat((data.test_mask, test_mask), dim=0)

    return data


def cat_tensor(pos_pairs):
    edge_index = pos_pairs[0]
    for edges in pos_pairs[1:]:
        edge_index = torch.cat((edge_index, edges), dim=1)
    return edge_index


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


class EMA:
    """
    Exponential Moving Average (EMA) is empirically shown to prevent the collapsing problem
    """
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new

        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)

    return 2 - 2 * (x * y).sum(dim=-1)


def id_loss_fn(x, y, pos_pairs, lambd, device):
    assert x.shape == y.shape
    loss = None
    x = (x - x.mean(0)) / x.std(0)
    y = (y - y.mean(0)) / y.std(0)
    for edge_index in pos_pairs:
        emb_1, emb_2 = x[edge_index[0]], y[edge_index[1]]
        N = emb_1.shape[0]
        c = torch.mm(emb_1.T, emb_2)
        c1 = torch.mm(emb_1.T, emb_1)
        c2 = torch.mm(emb_2.T, emb_2)

        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()
        l = loss_inv + lambd * (loss_dec1 + loss_dec2)
        loss = l if loss is None else loss+l
    return loss / len(pos_pairs)


def l2_normalize(x):
    return x / torch.sqrt(torch.sum(x**2, dim=1).unsqueeze(1))


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def repeat_1d_tensor(t, num_reps):
    return t.unsqueeze(1).expand(-1, num_reps)


def fill_ones(x):
    n_data = x.shape[0]
    x = torch.sparse_coo_tensor(x._indices(), torch.ones(x._nnz()).to(x.device), [n_data, n_data])
    return x


def random_aug(x, edge_index, feat_drop_rate, edge_mask_rate):
    x = drop_feature(x, feat_drop_rate)

    edge_mask = mask_edge(edge_index, edge_mask_rate)
    src = edge_index[0]
    dst = edge_index[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]
    edge_index = torch.stack((nsrc, ndst))

    return x, edge_index


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1),),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(edge_index, mask_prob):
    E = edge_index.shape[1]

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx
