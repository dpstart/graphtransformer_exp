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
        blocks = args.blocks

        num_layers = args.num_layers

        self.dropout = dropout
        self.num_classes = num_classes
        self.device = args.device

        self.layers = []

        self.embedding_enc = nn.Linear(pos_enc_dim, hidden_dim)
        self.embedding_h = nn.Linear(in_dim_node, hidden_dim)
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim, hidden_dim, num_heads, dropout, blocks=blocks
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim, out_dim, num_heads, dropout, blocks=blocks
            )
        )
        self.mlp = MLPReadout(out_dim, num_classes)

    def forward_bidirected(self, blocks, x, x_lap_pos_enc, src_nodes, dst_nodes):

        # EMBED + DROPOUT
        h = self.embedding_h(x)
        h = self.in_feat_dropout(h)

        # EMBED POSITIONAL ENCODINGS AND ADD
        h_lap_pos_enc = self.embedding_enc(x_lap_pos_enc.float())
        h = h + h_lap_pos_enc

        h_src = h[blocks[0].srcdata["_ID"]]

        # Iterate through transformer layers
        for i, layer in enumerate(self.layers):
            h_src = layer(blocks[i], h_src, h[blocks[i].dstdata["_ID"]])

        out = self.mlp(h_src)
        return out

    def forward(self, g, x, x_enc):
        # TODO add in feat dropout

        h = self.embedding_h(x)
        h = self.in_feat_dropout(h)
        h_enc = self.embedding_enc(x_enc)
        h = h + h_enc

        #h = self.dropout(h)

        # Iterate through transformer layers
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        out = self.mlp(h)
        return out

    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        return criterion(pred, label.squeeze())
