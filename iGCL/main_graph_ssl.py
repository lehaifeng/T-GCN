import torch
from torch_geometric.data import DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops, remove_self_loops
import torch_geometric.transforms as T
import numpy as np
import argparse

from src.data import Dataset
from src.configuration import Config
from src import utils
from src.IDGCL import IDGCL
from src.logreg import LogisticRegression

parser = argparse.ArgumentParser(description='SSL-Experiment')
parser.add_argument('--model', type=str, default='IDGCL', help='Name of model')
parser.add_argument('--root', type=str, default='./Data', help='Name of dataset.')
parser.add_argument('--dataset', type=str, default='WikiCS', help='Name of dataset.')
parser.add_argument('--dim', type=int, default=1024, help='Dimension of representations')
parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
parser.add_argument('--lambd', type=float, default=0.005, help='Weight of normalization loss')
parser.add_argument('--topk', type=int, default=6, help='Number of positive samples')
parser.add_argument('--epochs', type=int, default=1000, help="The maximum iterations of training")
parser.add_argument('--description', type=str, default=' ', help='Description of the experiment.')
command_args = parser.parse_args()


def infer_embeddings(model, loader, epoch, device):
    model.train(False)
    embeddings = labels = None
    for bc, batch_data in enumerate(loader):
        batch_data.to(device)
        emb = model.get_emb(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                            neighbor=[batch_data.neighbor_index, None],
                            edge_weight=None, epoch=epoch)
        emb = emb.detach()
        y = batch_data.y.detach()
        if embeddings is None:
            embeddings, labels = emb, y
        else:
            embeddings = torch.cat([embeddings, emb])
            labels = torch.cat([labels, y])
    return embeddings, labels


def evaluate_node(args, dataset, embeddings, labels, epoch, best_dev_acc, device, best_epoch, best_dev_std,
                  best_test_acc, best_test_std):
    emb_dim, num_class = embeddings.shape[1], labels.unique().shape[0]

    dev_accs, test_accs = [], []

    for i in range(20):

        train_mask = dataset[0].train_mask[i]
        dev_mask = dataset[0].val_mask[i]
        if args.dataset == "wikics":
            test_mask = dataset[0].test_mask
        else:
            test_mask = dataset[0].test_mask[i]

        classifier = LogisticRegression(emb_dim, num_class).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005, weight_decay=1e-4)  # default 0.01 1e-4

        for _ in range(1000):
            classifier.train()
            logits, loss = classifier(embeddings[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_logits, _ = classifier(embeddings[dev_mask], labels[dev_mask])
        test_logits, _ = classifier(embeddings[test_mask], labels[test_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        dev_acc = (torch.sum(dev_preds == labels[dev_mask]).float() /
                   labels[dev_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == labels[test_mask]).float() /
                    labels[test_mask].shape[0]).detach().cpu().numpy()

        dev_accs.append(dev_acc * 100)
        test_accs.append(test_acc * 100)

    dev_accs = np.stack(dev_accs)
    test_accs = np.stack(test_accs)

    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
    test_acc, test_std = test_accs.mean(), test_accs.std()

    print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(args.model, epoch, dev_acc,
                                                                                       dev_std, test_acc, test_std))

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_test_acc = test_acc
        best_dev_std = dev_std
        best_test_std = test_std
        best_epoch = epoch
        embeddings = embeddings.cpu().detach().numpy()
        file_path = r'./Embedding/{}_{}'.format(args.model, args.dataset)
        np.save(file_path, embeddings)

    # best_dev_accs.append(best_dev_acc)
    st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
        best_epoch, best_dev_acc, best_dev_std, best_test_acc, best_test_std)
    print(st_best)
    return best_epoch, best_dev_acc, best_dev_std, best_test_acc, best_test_std


def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    if args.dataset == 'cora':
        dataset = Planetoid(args.root, args.dataset, transform=T.NormalizeFeatures())
        dataset.data = utils.create_masks(dataset.data, args.dataset)
        dataset.data.neighbor_index, _ = remove_self_loops(dataset.data.edge_index)
        dataset.data.neighbor_attr = None
    else:
        dataset = Dataset(root=args.root, dataset=args.dataset)
        dataset.data.neighbor_index, _ = remove_self_loops(dataset.data.edge_index)

    loader = DataLoader(dataset=dataset)
    layers = [dataset.data.x.shape[1]] + args.hidden_layers

    model = IDGCL(layers, args).to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_test_acc, best_dev_acc, best_test_std, best_dev_std, best_epoch = 0, 0, 0, 0, 0

    # get Randomly Initialization accuracy
    embeddings, labels = infer_embeddings(model, loader, 0, args.device)
    print("Randomly Initialization Accuracy")
    best_epoch, best_dev_acc, best_dev_std, best_test_acc, best_test_std = evaluate_node(args, dataset,
                                                                                         embeddings, labels,
                                                                                         0, best_dev_acc,
                                                                                         device, best_epoch,
                                                                                         best_dev_std,
                                                                                         best_test_acc,
                                                                                         best_test_std)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for bc, batch_data in enumerate(loader):
            batch_data.to(args.device)
            _, loss = model(x=batch_data.x, y=batch_data.y, edge_index=batch_data.edge_index,
                            neighbor=[batch_data.neighbor_index, None],
                            edge_weight=None, epoch=epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_moving_average()

            st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(utils.currentTime(), epoch, args.epochs, loss.item())
            print(st)

        if epoch % args.eval_freq == 0:
            embeddings, labels = infer_embeddings(model, loader, 0, args.device)
            best_epoch, best_dev_acc, best_dev_std, best_test_acc, best_test_std = evaluate_node(args, dataset,
                                                                                                 embeddings, labels,
                                                                                                 epoch, best_dev_acc,
                                                                                                 device, best_epoch,
                                                                                                 best_dev_std,
                                                                                                 best_test_acc,
                                                                                                 best_test_std)

    print("\nTraining Done!")
    st_best = '[final]  ** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
        best_epoch, best_dev_acc, best_dev_std, best_test_acc, best_test_std)
    print(st_best)


if __name__ == '__main__':
    args = Config()

    args.model = command_args.model
    args.root = command_args.root
    args.dataset = command_args.dataset
    args.hidden_layers = [command_args.dim] * command_args.num_layers
    args.pred_hid = command_args.dim * 2
    args.topk = command_args.topk
    args.lambd = command_args.lambd
    args.epochs = command_args.epochs
    main(args)
