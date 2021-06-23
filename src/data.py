import dgl
import numpy as np

import random

from dgl.data import CoraGraphDataset, CiteseerGraphDataset

from ogb.nodeproppred import DglNodePropPredDataset


def get_dataset(dataset="cora", add_virtual_node=True):

    virtual_nodes = []
    if dataset == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        g, label = dataset[0]
        g.ndata["label"] = label

        num_classes = (np.amax(g.ndata["label"].numpy(), axis=0) + 1)[0]

    elif dataset == "citeseer":
        dataset = CiteseerGraphDataset()
        g = dataset[0]

        train_idx = np.nonzero(g.ndata["train_mask"]).squeeze()
        valid_idx = np.nonzero(g.ndata["val_mask"]).squeeze()
        test_idx = np.nonzero(g.ndata["test_mask"]).squeeze()
        num_classes = 6

    elif dataset == "cora":
        dataset = CoraGraphDataset()
        g = dataset[0]

        if add_virtual_node:

            num_real_nodes = g.num_nodes()
            num_virtual_nodes = 3
            real_nodes = list(range(num_real_nodes))
            g.add_nodes(num_virtual_nodes)

            # Change Topology
            virtual_src = []
            virtual_dst = []
            for count in range(num_virtual_nodes):
                virtual_node = num_real_nodes + count
                virtual_nodes.append(virtual_node)
                virtual_node_copy = [virtual_node] * num_real_nodes
                virtual_src.extend(real_nodes)
                virtual_src.extend(virtual_node_copy)
                virtual_dst.extend(virtual_node_copy)
                virtual_dst.extend(real_nodes)
                g.add_edges(virtual_src, virtual_dst)
                g.ndata["train_mask"][-num_virtual_nodes:] = 1

        # src_list = []
        # dst_list = []
        # for node1 in range(g.num_nodes()):
        #     for node2 in range(g.num_nodes() - 1):
        #         if random.uniform(0, 1) > 0.5:
        #             src_list.append(node1)
        #             dst_list.append(node2)

        # dgl.add_edges(g, src_list, dst_list)
        # g = dgl.to_simple(g)

        train_idx = np.nonzero(g.ndata["train_mask"]).squeeze()
        valid_idx = np.nonzero(g.ndata["val_mask"]).squeeze()
        test_idx = np.nonzero(g.ndata["test_mask"]).squeeze()
        num_classes = 7

    elif dataset == "products":
        dataset = DglNodePropPredDataset(name="ogbn-products")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        g, label = dataset[0]
        g.ndata["label"] = label

        num_classes = (np.amax(g.ndata["label"].numpy(), axis=0) + 1)[0]

    return (
        g,
        num_classes,
        train_idx,
        valid_idx,
        test_idx,
        virtual_nodes if len(virtual_nodes) > 0 else None,
    )
