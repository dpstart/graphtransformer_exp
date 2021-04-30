import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


import dgl
import dgl.function as fn
import numpy as np


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {
            out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(
                -1, keepdim=True
            )
        }

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


class MLPReadout(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.mlp(x)


class SubLayerWrapper(nn.Module):
    """
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    """

    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(F.relu(sublayer(self.norm(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.Q = nn.Linear(in_dim, out_dim * num_heads)
        self.K = nn.Linear(in_dim, out_dim * num_heads)
        self.V = nn.Linear(in_dim, out_dim * num_heads)

    def propagate(self, g):

        # update the features of the specified edges by the provided function
        # In this case, each edge feature is updated with the attention score
        # between the edge nodes.
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))

        # Exp and scale by constant
        g.apply_edges(scaled_exp("score", np.sqrt(self.out_dim)))

        eids = g.edges()

        # Value weighted by attention scores.
        g.send_and_recv(
            eids, fn.src_mul_edge("V_h", "score", "V_h"), fn.sum("V_h", "wV")
        )

        # Normalization value for the softmax. Each nodes receives the
        # attention score from the neighbors, and they are summed.
        g.send_and_recv(eids, fn.copy_edge("score", "score"), fn.sum("score", "z"))

    def forward(self, g, x):

        Q_h = self.Q(x)
        K_h = self.K(x)
        V_h = self.V(x)

        g.ndata["Q_h"] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["K_h"] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata["V_h"] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate(g)
        out = g.ndata["wV"]  # / g.ndata["z"]
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0):

        super(GraphTransformerLayer, self).__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiHeadAttention(in_dim, out_dim // num_heads, num_heads)
        self.O = nn.Linear(out_dim, out_dim)
        self.sublayer = SubLayerWrapper(out_dim, dropout)

    def forward(self, g, x):

        h_in1 = x
        attn_out = self.attention(g, x)

        h = h_in1 + (
            F.dropout(
                attn_out.view(-1, self.out_channels),
                self.dropout,
                training=self.training,
            )
        )

        h = self.norm(h)
        h = self.sublayer(h, self.O)

        return h
