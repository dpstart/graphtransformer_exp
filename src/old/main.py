import torch
import torch.nn as nn
from scipy import sparse as sp
import numpy as np

import dgl
from dgl.data import CoraGraphDataset

from ogb.nodeproppred import DglNodePropPredDataset

from model import GraphTransformer
from train import train_iter
from args import parse_args

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def accuracy(scores, targets):
    scores = scores.argmax(dim=1)
    acc = (scores == targets.squeeze()).sum()
    acc = acc / len(targets)
    return acc


def add_encodings(g, dim, type="lap"):

    if type == "lap":
        return add_lap_encodings(g, dim)
    elif type == "dummy":
        A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
        g.ndata["lap_pos_enc"] = torch.zeros((A.shape[0], dim)).float()
        return g


def add_lap_encodings(g, dim):
    """Add Laplacian positional encodings to the graph.
    """

    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    eigval, eigvec = sp.linalg.eigs(L, k=dim + 1, which="SR", tol=1e-2)

    eigvec = eigvec[:, eigval.argsort()]
    g.ndata["lap_pos_enc"] = torch.from_numpy(eigvec[:, 1 : dim + 1]).float()
    return g


def flip(lap_pos_enc):
    sign_flip = torch.rand(lap_pos_enc.size(1))
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    return lap_pos_enc * sign_flip.unsqueeze(0)


def run_single_graph(g, args, *idx):
    """Run training on a single graph.
    """

    train_idx, val_idx, test_idx = idx

    model = GraphTransformer(args).to(args.device)
    model.apply(init_weights)
    print(f"[!] No. of params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    g_ = g.clone()
    x = g.ndata["feat"].to(args.device)
    lap_pos_enc = flip(g.ndata["lap_pos_enc"]).to(args.device)
    labels = g.ndata["label"].to(args.device)

    for epoch in range(args.epochs):

        g = g_.to(args.device)

        model.train()
        optimizer.zero_grad()

        scores = model(g, x, lap_pos_enc)

        loss = model.loss(scores[train_idx], labels.squeeze()[train_idx])
        loss.backward()
        optimizer.step()
        acc = accuracy(scores.detach()[train_idx], labels.detach()[train_idx])

        train_losses.append(loss.detach().item())
        train_accs.append(acc)
        print(f"Epoch: {epoch} | Train Loss: {loss:.4f} | Train Acc: {acc:.4f}")

        if epoch % args.val_interval == 0:

            model.eval()

            with torch.no_grad():
                scores = model.forward(g, x, lap_pos_enc)
                loss = model.loss(scores[val_idx], labels.squeeze()[val_idx])

            acc = accuracy(scores[val_idx], labels[val_idx])

            print(f"Epoch: {epoch} | Val Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    # TEST
    model.eval()

    with torch.no_grad():
        scores = model.forward(g, x, lap_pos_enc)
        loss = model.loss(scores[test_idx], labels.squeeze()[test_idx])

    acc = accuracy(scores[test_idx], labels[test_idx])

    print(f"Epoch: {epoch} | Test Loss: {loss:.4f} | Test Acc: {acc:.4f}")


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

    elif args.dataset == "products":
        dataset = DglNodePropPredDataset(name="ogbn-products")
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = (
            split_idx["train"],
            split_idx["valid"],
            split_idx["test"],
        )
        g, label = dataset[0]
        g.ndata["label"] = label

        args.num_classes = (np.amax(g.ndata["label"].numpy(), axis=0) + 1)[0]

    args.in_dim = g.ndata["feat"].shape[1]

    print("[!] Dataset loaded")
    print(
        f"[!] No. of nodes: {g.num_nodes()} | No. of feature dimensions: {args.in_dim} | No. of classes: {args.num_classes}"
    )

    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    g = add_encodings(g, int(args.pos_enc_dim), type="lap")
    print("[!] Added positional encodings")

    run_single_graph(g, args, train_idx, valid_idx, test_idx)


if __name__ == "__main__":
    main()
