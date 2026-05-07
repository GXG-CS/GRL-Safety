"""GNN baseline (from-scratch) module.

Provides a unified encoder wrapper over GCN/GAT/GraphSAGE that matches the
BGRL/GraphMAE encoder interface:
  - forward(data) → [N, hidden] node embeddings
  - .representation_size attribute for downstream head construction

All three encoders use uniform FT hyperparameters (no per-dataset tuning):
  lr=1e-3, dropout=0.2, weight_decay=1e-4, max_epochs=500, patience=200,
  hidden=768, num_layers=2
"""

import torch
import torch.nn as nn

from sgb.models.gcn import GCN
from sgb.models.gat import GAT
from sgb.models.sage import GraphSAGE


class GNNEncoderWrapper(nn.Module):
    """Wraps GCN/GAT/SAGE to match the data-object forward used by FT scripts.

    Matches BGRL encoder's interface:
      - forward(data) -> [N, hidden]
      - .representation_size for head

    Call with `model_name` in {"gcn", "gat", "sage"}.
    """

    def __init__(self, model_name: str, in_channels: int,
                 hidden_channels: int = 768, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.model_name = model_name.lower()
        if self.model_name == "gcn":
            self.encoder = GCN(in_channels, hidden_channels,
                               num_layers=num_layers, dropout=dropout)
        elif self.model_name == "gat":
            self.encoder = GAT(in_channels, hidden_channels,
                               num_layers=num_layers, dropout=dropout)
        elif self.model_name == "sage":
            self.encoder = GraphSAGE(in_channels, hidden_channels,
                                     num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(
                f"Unknown model_name {model_name!r}; expected gcn/gat/sage")
        self.representation_size = hidden_channels

    def forward(self, data):
        # data has .x and .edge_index; we ignore edge_attr (none of the three
        # baseline encoders consume it in our setup)
        return self.encoder(data.x, data.edge_index)


METHOD_NAMES = {
    "gcn": "GCN",
    "gat": "GAT",
    "sage": "SAGE",
}
