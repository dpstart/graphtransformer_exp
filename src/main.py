import torch

import dgl
from dgl.data import CoraGraphDataset

from ogb.nodeproppred import DglNodePropPredDataset

from model import GraphTransformer
from train import train_iter, evaluate, train_iter_batched, evaluate_batched
from args import parse_args

from scipy import sparse as sp

import numpy as np


def add_encodings(g, dim, type="lap"):

    if type == "lap":
        return add_lap_encodings(g, dim)
    elif type == "dummy":
        A = g.adjacency_matrix(scipy_fmt="csr").astype(float)

        g.ndata["lap_pos_enc"] = torch.zeros((A.shape[0], dim)).float()
        return g


def add_lap_encodings(g, dim):

    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    eigval, eigvec = sp.linalg.eigs(L, k=dim + 1, which="SR", tol=1e-2)

    eigvec = eigvec[:, eigval.argsort()]
    g.ndata["lap_pos_enc"] = torch.from_numpy(eigvec[:, 1 : dim + 1]).float()
    return g


def run_single_graph_batched(args, g, *idx):

    train_idx, valid_idx, test_idx = idx

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_idx,
        sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        valid_idx,
        sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_idx,
        sampler,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        num_workers=1,
    )

    model = GraphTransformer(args)

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

        for input_nodes, output_nodes, blocks in train_dataloader:

            loss, acc, optimizer = train_iter_batched(
                model,
                g,
                input_nodes,
                output_nodes,
                blocks,
                optimizer,
                args.device,
                epoch,
            )

            epoch_train_losses.append(loss)
            epoch_train_accs.append(acc)
        print(
            f"Epoch: {epoch} | Train Loss: {np.mean(epoch_train_losses):.4f} | Train Acc: {np.mean(epoch_train_accs):.4f}"
        )

        ### Validation
        if epoch % 10 == 0:

            epoch_val_losses, epoch_val_accs = [], []

            for input_nodes, output_nodes, blocks in val_dataloader:

                eval_loss, eval_acc = evaluate_batched(
                    model, g, input_nodes, output_nodes, blocks, args.device
                )
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
        f"Epoch: {epoch} | Test Loss: {np.mean(test_losses):.4f} | Eval Acc: {np.mean(test_accs):.4f}"
    )


def run_single_graph(args, g, *a):

    train_idx, valid_idx, test_idx = a

    model = GraphTransformer(args)

    print(f"[!] Number of params: {sum(p.numel() for p in model.parameters())}")
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
        loss, acc, optimizer = train_iter(
            model, g, train_idx, optimizer, args.device, epoch
        )

        if epoch % 10 == 0:
            eval_loss, eval_acc = evaluate(model, g, val_idx, args.device)
            print(
                f"Epoch: {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.4f} | Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}"
            )
            scheduler.step(eval_loss)
    test_loss, test_acc = evaluate(model, g, g.ndata["test_mask"], args.device)
    print(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")


def run_multiple_graphs(args, dataloader):
    pass


def main():

    args = parse_args()

    if args.dataset == "arxiv":
        dataset = DglNodePropPredDataset(name="ogbn-arxiv")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        g, label = dataset[0]
        g.ndata["label"] = label

        args.num_classes = (np.amax(g.ndata["label"].numpy(), axis=0) + 1)[0]

    elif args.dataset == "cora":

        dataset = CoraGraphDataset()
        g = dataset[0]

        train_idx = np.nonzero(g.ndata["train_mask"]).squeeze()
        valid_idx = np.nonzero(g.ndata["val_mask"]).squeeze()
        test_idx = np.nonzero(g.ndata["test_mask"]).squeeze()
        args.num_classes = 7

    args.in_dim = g.ndata["feat"].shape[1]

    print("[!] Dataset loaded")
    print(
        f"[!] No. of nodes: {g.num_nodes()} | No. of feature dimensions: {args.in_dim} | No. of classes: {args.num_classes}"
    )

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    g = add_encodings(g, int(args.pos_enc_dim), type="lap")
    print("[!] Added positional encodings")
    run_single_graph_batched(args, g, train_idx, valid_idx, test_idx)


if __name__ == "__main__":
    main()
