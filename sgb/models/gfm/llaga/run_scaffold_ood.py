"""LLaGA scaffold-OOD on BBBP / BACE / Tox21.

Architecture:
  SBERT atom features [N, 768]
    -> LLaGA pretrained 2-layer MLP projector (frozen, gft9 ckpt) [N, 4096]
    -> trainable compress Linear (4096 -> hidden=512)
    -> 2-layer GIN over mol bonds (hidden -> hidden, BatchNorm, ReLU)
    -> mean-pool atoms within each mol -> [B, hidden]
    -> linear classifier head -> [B, num_classes]

Trainable: compress + GIN + head. Projector frozen (carries gft9 prior).
Same protocol/runner as scaffold_ood_runner.
"""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.scaffold_ood_runner import run_scaffold_ood


def _build_projector(in_dim=768, hidden=4096, depth=2):
    """LLaGA's k-layer-mlp projector (matches model._build_projector)."""
    mods = [nn.Linear(in_dim, hidden)]
    for _ in range(1, depth):
        mods.append(nn.GELU())
        mods.append(nn.Linear(hidden, hidden))
    return nn.Sequential(*mods)


class _GINLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, x[src])
        h = (1 + self.eps) * x + agg
        h = self.mlp(h)
        h = self.bn(h)
        return F.relu(h)


def _scatter_mean(x, idx, num_groups):
    """Mean over rows grouped by idx. Returns [num_groups, D]."""
    out = x.new_zeros(num_groups, x.size(1))
    counts = x.new_zeros(num_groups, 1)
    out.index_add_(0, idx, x)
    counts.index_add_(0, idx, torch.ones_like(x[:, :1]))
    return out / counts.clamp_min(1.0)


class _LlagaScaffoldLite(nn.Module):
    def __init__(self, num_classes, hidden=512, dropout=0.2,
                 projector_ckpt=None, n_gin_layers=2):
        super().__init__()
        self.projector = _build_projector(in_dim=768, hidden=4096, depth=2)
        if projector_ckpt:
            pt = torch.load(projector_ckpt, map_location="cpu", weights_only=False)
            self.projector.load_state_dict(pt, strict=True)
            print(f"[LLaGA] loaded projector from {projector_ckpt}")
        # freeze projector to keep gft9 pretraining prior
        for p in self.projector.parameters():
            p.requires_grad = False

        self.compress = nn.Sequential(
            nn.Linear(4096, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.gin = nn.ModuleList(
            [_GINLayer(hidden, dropout=dropout) for _ in range(n_gin_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, batch):
        # batch.x [N_atoms, 768] SBERT features
        with torch.no_grad():
            h = self.projector(batch.x)  # frozen; no grad needed
        h = self.compress(h)              # trainable
        for layer in self.gin:
            h = layer(h, batch.edge_index)
        B = int(batch.batch.max().item()) + 1
        h = _scatter_mean(h, batch.batch, B)
        h = self.dropout(h)
        return self.head(h)


_ARGS = None


def _builder(in_channels, num_tasks, dropout, device):
    model = _LlagaScaffoldLite(
        num_classes=num_tasks,
        hidden=_ARGS.hidden,
        dropout=dropout,
        projector_ckpt=_ARGS.projector_ckpt,
        n_gin_layers=_ARGS.n_gin_layers,
    )
    return model.to(device)


def main():
    global _ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--projector_ckpt",
                   default="ckpts/LLaGA/projector_gft9_vicuna.pt")
    p.add_argument("--max_epochs", default=200, type=int)
    p.add_argument("--patience", default=40, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.2, type=float)
    p.add_argument("--n_seeds", default=3, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--hidden", default=512, type=int)
    p.add_argument("--n_gin_layers", default=2, type=int)
    _ARGS = p.parse_args()

    run_scaffold_ood(
        method_tag="LLaGA",
        dataset=_ARGS.dataset,
        build_ft_model=_builder,
        lr=_ARGS.lr,
        weight_decay=_ARGS.wd,
        dropout=_ARGS.dropout,
        max_epochs=_ARGS.max_epochs,
        patience=_ARGS.patience,
        n_seeds=_ARGS.n_seeds,
        batch_size=_ARGS.batch_size,
    )


if __name__ == "__main__":
    main()
