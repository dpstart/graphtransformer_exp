import torch
import torch.nn as nn
from scipy import sparse as sp
import numpy as np

import dgl

from model import GraphTransformer
from train import train_iter_batched, evaluate_batched
from args import get_parser
from util import init_weights, print_args, get_dataloaders, all_pairs_sp
from data import get_dataset

import argparse
import json

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def add_encodings(g, dim, type="lap"):

    if type == "lap":
        return add_lap_encodings(g, dim)
    elif type == "dummy":
        A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
        g.ndata["lap_pos_enc"] = torch.zeros((A.shape[0], dim)).float()
        return g


def add_lap_encodings(g, dim):
    """Add Laplacian positional encodings to the graph."""

    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    eigval, eigvec = sp.linalg.eigs(L, k=dim + 1, which="SR", tol=1e-2)

    eigvec = eigvec[:, eigval.argsort()]
    g.ndata["lap_pos_enc"] = torch.from_numpy(eigvec[:, 1 : dim + 1]).float()
    return g


def init_params(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


def run_single_graph_batched(g, args, *idx, virtual_nodes=None):
    """Run training on a single graph, using minibatches for the nodes."""

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        g, args, *idx, virtual_nodes=virtual_nodes
    )

    model = GraphTransformer(args)
    model = init_params(model)
    print(f"[!] No. of params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_reduce_factor,
        patience=args.lr_schedule_patience,
    )

    for epoch in range(args.epochs):

        epoch_train_losses, epoch_val_losses = [], []
        epoch_train_accs, epoch_val_accs = [], []

        for blocks in train_dataloader:

            loss, acc, optimizer = train_iter_batched(
                model, g, blocks, optimizer, args.device
            )

            epoch_train_losses.append(loss)
            epoch_train_accs.append(acc)
        print(
            f"Epoch: {epoch} | Train Loss: {np.mean(epoch_train_losses):.4f} | Train Acc: {np.mean(epoch_train_accs):.4f}"
        )

        ### Validation
        if epoch % args.val_interval == 0:

            epoch_val_losses, epoch_val_accs = [], []
            for _, _, blocks in val_dataloader:

                eval_loss, eval_acc = evaluate_batched(model, g, blocks, args.device)
                epoch_val_losses.append(eval_loss)
                epoch_val_accs.append(eval_acc)

            print(
                f"Epoch: {epoch} | Val Loss: {np.mean(epoch_val_losses):.4f} | Eval Acc: {np.mean(epoch_val_accs):.4f}"
            )
            scheduler.step(np.mean(epoch_val_losses))

    test_losses, test_accs = [], []

    for input_nodes, output_nodes, blocks in test_dataloader:

        test_loss, test_acc = evaluate_batched(
            model, g, input_nodes, output_nodes, blocks, args.device
        )
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print(
        f"Epoch: {epoch} | Test Loss: {np.mean(test_losses):.4f} | Test Acc: {np.mean(test_accs):.4f}"
    )


def main():

    parser = get_parser()
    args = parser.parse_args()
    args.blocks = True

    if args.config:
        with open(args.config, "rt") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    print_args(args)

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ###### Load Datasets

    g, num_classes, train_idx, valid_idx, test_idx, virtual_nodes = get_dataset(
        args.dataset, add_virtual_node=True
    )

    # All pairs shortest path distance
    # sp = all_pairs_sp(g)

    # dist = []
    # close = {}
    # n = 5
    # for n in range(g.num_nodes()):

    #     for k, v in sp[n].items():
    #         dist.append(v)

    #     for i, n in enumerate(dist):
    #         if n == 1:
    #             dist[i] = 9999999999
    #     close[k] = np.argsort(dist)[:n]
    #     dist = []

    # for k, v in close.items():
    #     g.add_edges([k] * len(v), v)

    # g = dgl.add_reverse_edges(g)
    # g = dgl.to_simple(g)

    args.num_classes = num_classes
    args.in_dim = g.ndata["feat"].shape[1]

    print("[!] Dataset loaded")
    print(
        f"[!] No. of nodes: {g.num_nodes()} | No. of feature dimensions: {args.in_dim} | No. of classes: {args.num_classes}"
    )

    #### Add Positional Encodings
    g = add_encodings(g, int(args.pos_enc_dim), type="lap")
    print("[!] Added positional encodings")
    run_single_graph_batched(
        g, args, train_idx, valid_idx, test_idx, virtual_nodes=virtual_nodes
    )


if __name__ == "__main__":
    main()
