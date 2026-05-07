"""GAT encoder.

Uses PyG built-in GATConv. Outputs [N, hidden_channels] embedding.
No classification head — that's in eval/heads.py (shared by all methods).
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.5,
                 heads: int = 8, out_heads: int = 1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=out_heads, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
