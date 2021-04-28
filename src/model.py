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

        self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim, hidden_dim, num_heads, dropout, args.small
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, num_heads, dropout, args.small)
        )
        self.mlp = MLPReadout(out_dim, num_classes)

    def forward(self, blocks, x, x_lap_pos_enc, src_nodes, dst_nodes):
        # TODO add in feat dropout

        h = self.embedding_h(x)
        h = F.dropout(h, self.dropout, training=self.training)
        h_lap_pos_enc = self.embedding_lap_pos_enc(x_lap_pos_enc.float())
        h = h + h_lap_pos_enc

        h_src = h[blocks[0].srcdata["_ID"]]

        # Iterate through transformer layers
        for i, layer in enumerate(self.layers):
            h_src = layer(blocks[i], h_src, h[blocks[i].dstdata["_ID"]])

        out = self.mlp(h_src)
        return out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label.squeeze())
