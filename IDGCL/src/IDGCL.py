import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj
from torch_cluster import random_walk
import copy
import numpy as np
from src import utils


class IDGCL(nn.Module):
    def __init__(self, layer_config, args, **kwargs):
        super(IDGCL, self).__init__()
        self.student_encoder = Encoder(layer_config, dropout=args.dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        utils.set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = utils.EMA(args.mad, args.epochs)

        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, args.pred_hid), nn.BatchNorm1d(args.pred_hid),
                                               nn.PReLU(),
                                               nn.Linear(args.pred_hid, rep_dim))
        self.student_predictor.apply(utils.init_weights)

        self.topk = args.topk
        self.loss_fn = args.loss_fn
        self.aug_way = args.aug_way
        self.device = args.device
        self.lambd = args.lambd

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        utils.update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def create_sparse(self, I):
        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)

        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def augment_adj(self, adj, student, teacher, top_k):
        n_data, d = student.shape
        pos_pairs = [torch.tensor([list(range(n_data)), list(range(n_data))]).to(self.device)]
        if top_k == 1:
            return pos_pairs, top_k

        similarity = torch.sum(student[adj.coalesce()._indices()[0]] * teacher[adj.coalesce()._indices()[1]],
                               dim=-1).squeeze()  # adj includes self-loop
        similarity = torch.sparse_coo_tensor(adj.coalesce()._indices(), similarity, (n_data, n_data)).to_dense()
        _, I_knn = similarity.topk(k=top_k - 1, dim=1, largest=True, sorted=True)
        for i in range(top_k - 1):
            cur_knn = I_knn[:, i].unsqueeze(1)
            knn_neighbor = self.create_sparse(cur_knn)
            locality = knn_neighbor * adj
            pos_pairs.append(locality.coalesce()._indices())

        return pos_pairs, top_k

    def augment_random_walk(self, edge_index, student, teacher, top_k):
        # random walk
        walk_length = 30
        N = student.shape[0]
        start = torch.Tensor(range(N)).long().to(self.device)

        walk = random_walk(edge_index[0], edge_index[1], start, walk_length=walk_length, p=1000)[:, 1:]
        nodes = torch.arange(N).int().reshape(N, 1)
        target_nodes = torch.cat((nodes, nodes), dim=-1)
        for _ in range(walk_length - 2):
            target_nodes = torch.cat((target_nodes, nodes), dim=-1)
        target_nodes = target_nodes.view(-1).long().to(self.device)
        source_nodes = walk.reshape(N * walk_length).long()
        # calculate similarity
        sim = torch.sum(student[target_nodes] * teacher[source_nodes], dim=1).view(-1).reshape((N, walk_length))
        _, knn_idx = sim.topk(k=top_k, dim=1, largest=True, sorted=True)
        walk = walk.to(self.device)
        pos_idx = walk.gather(1, knn_idx[:, 0].reshape((N, 1)))
        for i in range(1, top_k):
            pos_idx = torch.cat((pos_idx, walk.gather(1, knn_idx[:, i].reshape((N, 1)))), dim=-1)
        target_nodes = target_nodes.reshape((N, walk_length))[:, :top_k]
        target_nodes = target_nodes.reshape((N * top_k)).squeeze()
        locality = torch.stack((target_nodes, pos_idx.view(-1)), dim=0)
        return locality, top_k

    def forward(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        pred = self.student_predictor(student)
        with torch.no_grad():
            teacher = self.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)

        if edge_weight == None:
            adj = torch.sparse.FloatTensor(neighbor[0], torch.ones_like(neighbor[0][0]), [x.shape[0], x.shape[0]])
        else:
            adj = torch.sparse.FloatTensor(neighbor[0], neighbor[1], [x.shape[0], x.shape[0]])

        pos_pairs, k = self.augment_adj(adj, F.normalize(student, dim=-1, p=2), F.normalize(teacher, dim=-1, p=2),
                                    self.topk)

        loss = utils.id_loss_fn(pred, teacher, pos_pairs, self.lambd, self.device)
        return student, loss

    def get_emb(self, x, y, edge_index, neighbor, edge_weight=None, epoch=None):
        student = self.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        return student


class Encoder(nn.Module):
    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super(Encoder, self).__init__()
        self.stacked_gnn = nn.ModuleList(
            [GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_acts = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index, edge_weight=None):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=edge_weight)
            x = self.stacked_bns[i](x)
            x = self.stacked_acts[i](x)

        return x