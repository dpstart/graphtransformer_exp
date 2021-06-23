import os

import torch

from partition_util import *


class NeighborSampler(object):
    def __init__(self, g, fanouts, add_nodes):
        self.g = g
        self.fanouts = fanouts
        self.add_nodes = add_nodes

    def sample_blocks(self, seeds):

        if self.add_nodes is not None:
            seeds = np.append(seeds, self.add_nodes)
        seeds = torch.LongTensor(seeds)
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, np.unique(seeds), fanout)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, np.unique(seeds))
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks


class ClusterIter(object):
    """The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    """

    def __init__(self, dn, g, psize, batch_size):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        batch_size: int
            The number of partitions in one batch
        """
        self.psize = psize
        self.batch_size = batch_size
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join("./datasets/", dn + "_{}.npy".format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs("./datasets/", exist_ok=True)
                self.par_li = get_partition_list(g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(g, psize)
        par_list = []
        for p in self.par_li:
            par = torch.Tensor(p)
            par_list.append(par)
        self.par_list = par_list

    def __len__(self):
        return self.psize

    def __getitem__(self, idx):
        return self.par_li[idx]


def subgraph_collate_fn(g, batch):
    nids = np.concatenate(batch).reshape(-1).astype(np.int64)
    g1 = g.subgraph(nids)
    g1 = dgl.remove_self_loop(g1)
    g1 = dgl.add_self_loop(g1)
    return g1
