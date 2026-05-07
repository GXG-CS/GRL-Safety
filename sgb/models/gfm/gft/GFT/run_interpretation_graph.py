"""GFT FT + graph-level interpretation eval (Tox21 / ChemHIV).

Bypasses GFT's TaskModel + VQ + proto path; uses GFT's pretrained
*encoder* as a feature extractor with global_mean_pool + linear head,
on equal footing with BGRL/GraphMAE/UG2/OFA wrappers.

We benchmark GFT's *learned representation* under a standard graph-cls
FT, not GFT's full prompt-tree formulation (would require a separate
pipeline). Same convention used in OFA's graph-interp wrapper.
"""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", "..", ".."))
for p in (_HERE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.encoder import Encoder  # noqa: E402
from sgb.data.graph_interpretation_runner import run_graph_interpretation  # noqa: E402


_CKPT_DIR = None


class _FTGraphModel(nn.Module):
    def __init__(self, encoder, num_classes, hidden_dim, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool
        # GFT Encoder forward: encode(x, edge_index, edge_attr=None)
        h = self.encoder(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)
        h = self.dropout(h)
        return self.head(h)


def _builder(in_channels, num_tasks, dropout, device):
    enc = Encoder(
        input_dim=in_channels,
        hidden_dim=768,
        activation=nn.ReLU,
        num_layers=2,
        backbone='sage',
        normalize='none',
        dropout=dropout,
    )
    if _CKPT_DIR is not None:
        ck_path = osp.join(_CKPT_DIR, "encoder.pt")
        if osp.exists(ck_path):
            state = torch.load(ck_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            res = enc.load_state_dict(state, strict=False)
            print(f"[GFT-GINTERP] loaded encoder.pt: missing/unexpected={res}", flush=True)
        else:
            print(f"[GFT-GINTERP] WARN: ckpt path not found: {ck_path}", flush=True)
    return _FTGraphModel(enc, num_classes=num_tasks, hidden_dim=768, dropout=dropout).to(device)


def main():
    global _CKPT_DIR
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt_dir", default=osp.abspath(osp.join(
        _HERE, "..", "..", "..", "..", "ckpts", "GFT")))
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_seeds", type=int, default=5)
    args = p.parse_args()
    _CKPT_DIR = args.ckpt_dir
    print(f"[GFT-GINTERP] dataset={args.dataset} ckpt_dir={_CKPT_DIR} seeds={args.n_seeds}", flush=True)

    run_graph_interpretation(
        method_tag="GFT",
        dataset=args.dataset,
        build_ft_model=_builder,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    main()
