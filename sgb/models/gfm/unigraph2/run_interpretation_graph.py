"""UniGraph2 FT + scaffold-OOD eval on BBBP / BACE."""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn
import dgl

_UG2_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_UG2_DIR, "..", "..", ".."))
if _UG2_DIR not in sys.path:
    sys.path.insert(0, _UG2_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.gfm.unigraph2.run_feature_noise import build_model
from sgb.data.graph_interpretation_runner import run_graph_interpretation


class _FTGraphModel(nn.Module):
    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.2):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool
        src, dst = batch.edge_index[0], batch.edge_index[1]
        g = dgl.graph((src, dst), num_nodes=batch.x.size(0))
        g = g.remove_self_loop().add_self_loop().to(batch.x.device)
        h = self.pre_model(g, {"text": batch.x}, spd_matrix=None, return_embeddings=True)
        h = global_mean_pool(h, batch.batch)
        h = self.dropout(h)
        return self.head(h)


_CKPT_STATE = None
_ARGS = None


def _builder(in_channels, num_tasks, dropout, device):
    global _CKPT_STATE
    if _CKPT_STATE is None:
        ckpt = torch.load(_ARGS.ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        _CKPT_STATE = ckpt
    pre_model = build_model(num_features=in_channels)
    pre_model.load_state_dict(_CKPT_STATE, strict=False)
    return _FTGraphModel(pre_model, num_hidden=in_channels, num_classes=num_tasks,
                         dropout=dropout).to(device)


def main():
    global _ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--patience", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--n_seeds", default=5, type=int)
    _ARGS = parser.parse_args()

    run_graph_interpretation(
        method_tag="UniGraph2_FT",
        dataset=_ARGS.dataset,
        build_ft_model=_builder,
        lr=_ARGS.lr,
        weight_decay=_ARGS.wd,
        dropout=_ARGS.dropout,
        max_epochs=_ARGS.max_epochs,
        patience=_ARGS.patience,
        n_seeds=_ARGS.n_seeds,
    )


if __name__ == "__main__":
    main()
