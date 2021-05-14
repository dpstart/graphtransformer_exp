import torch
import dgl


import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops


def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = np.mean(matching)
    return edge_hom


def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node, targ_node = A.nonzero()
    edge_idx = torch.tensor(
        np.vstack((src_node, targ_node)), dtype=torch.long
    ).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def print_args(args):
    args_dict = vars(args)
    print("+------------------------------------------------------+")
    for k, v in args_dict.items():
        print("|", k, v)
    print("+------------------------------------------------------+")


def get_dataloaders(g, args, *idx):

    train_idx, valid_idx, test_idx = idx

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        valid_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_idx,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    return train_dataloader, val_dataloader, test_dataloader
