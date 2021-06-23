import torch
import dgl

import networkx as nx


import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import remove_self_loops
from sampler import NeighborSampler


def all_pairs_sp(g):
    path = dict(nx.all_pairs_shortest_path_length(g.to_networkx()))
    return path


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


def get_dataloaders(g, args, *idx, virtual_nodes=None):

    train_idx, valid_idx, test_idx = idx

    # sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layers)
    sampler_val = dgl.dataloading.MultiLayerNeighborSampler(
        [10 * i for i in range(1, args.num_layers + 1)]
    )
    # train_dataloader = dgl.dataloading.NodeDataLoader(
    #     g,
    #     train_idx,
    #     sampler,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     drop_last=False,
    #     num_workers=args.num_workers,
    # )

    sampler = NeighborSampler(
        g, [10 * i for i in range(1, args.num_layers + 1)], add_nodes=virtual_nodes
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_idx,
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        valid_idx,
        sampler_val,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_idx,
        sampler_val,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    return train_dataloader, val_dataloader, test_dataloader


def make_full_graph(g):

    full_g = dgl.from_networkx(nx.complete_graph(g.number_of_nodes()))

    # Here we copy over the node feature data and laplace encodings
    full_g.ndata["feat"] = g.ndata["feat"]

    try:
        full_g.ndata["EigVecs"] = g.ndata["EigVecs"]
        full_g.ndata["EigVals"] = g.ndata["EigVals"]
    except:
        pass

    # Populate edge features w/ 0s
    full_g.edata["feat"] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)
    full_g.edata["real"] = torch.zeros(full_g.number_of_edges(), dtype=torch.long)

    # Copy real edge data over
    # full_g.edges[g.edges(form="uv")[0].tolist(), g.edges(form="uv")[1].tolist()].data[
    #     "feat"
    # ] = torch.ones(g.edata["feat"].shape[0], dtype=torch.long)
    # full_g.edges[g.edges(form="uv")[0].tolist(), g.edges(form="uv")[1].tolist()].data[
    #     "real"
    # ] = torch.ones(g.edata["feat"].shape[0], dtype=torch.long)

    return full_g
