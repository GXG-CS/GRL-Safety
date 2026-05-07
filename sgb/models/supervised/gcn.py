"""GCN encoder.

Uses PyG built-in GCNConv. Outputs [N, hidden_channels] embedding.
No classification head — that's in eval/heads.py (shared by all methods).
Also serves as the reference model for Corruption mCE normalization.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
