import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl

from layers import GraphTransformerLayer, MLPReadout


class GraphTransformer(nn.Module):
    def __init__(self, args):

        super().__init__()

        in_dim_node = args.in_dim
        hidden_dim = args.hidden_dim
        out_dim = args.out_dim
        num_classes = args.num_classes
        num_heads = args.num_heads
        in_feat_dropout = args.in_feat_dropout
        dropout = args.dropout
        pos_enc_dim = args.pos_enc_dim

        num_layers = args.num_layers

        self.dropout = dropout
        self.num_classes = num_classes
        self.device = args.device
        self.lap_pos_enc = args.lap_pos_enc

        self.layers = []

        self.embedding_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(hidden_dim, hidden_dim, num_heads, dropout)
                for _ in range(num_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout)
        )

        self.mlp = MLPReadout(out_dim, num_classes)

    def forward(self, g, x, x_enc):
        # TODO add in feat dropout

        h = self.embedding_h(x)
        h_enc = self.embedding_enc(x_enc)
        h = h + h_enc

        h = self.dropout(h)

        # Iterate through transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        out = self.mlp(h)
        return out

    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        # V = label.size(0)
        # label_count = torch.bincount(label)
        # label_count = label_count[label_count.nonzero()].squeeze()
        # cluster_sizes = torch.zeros(self.num_classes).long().to(self.device)
        # cluster_sizes[torch.unique(label)] = label_count
        # weight = (V - cluster_sizes).float() / V
        # weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)

        return loss

