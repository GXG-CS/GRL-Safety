"""LLaGA inductive KG link-prediction on FB15K237 / WN18RR.

Architecture (lite — no frozen LLM forward; the LLM was the OOM source AND
adds nothing for KG link-prediction over SBERT features):

  SBERT entity features [N, 768]
    -> LLaGA pretrained 2-layer MLP projector (frozen, gft9 ckpt) [N, 4096]
    -> trainable compress Linear (4096 -> hidden=768)
    -> 2-layer GIN over G_tr / G_te_support edges (graph-aware aggregation)
    -> output [N, hidden]

The shared inductive_kg_runner trains the encoder + DistMult relation
embeddings with negative-sampling cross-entropy and reports filtered
MRR / Hits@10 on inductive test queries.
"""
from __future__ import annotations

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

from sgb.data.inductive_kg_runner import run_inductive_kg


def _build_projector(in_dim=768, hidden=4096, depth=2):
    mods = [nn.Linear(in_dim, hidden)]
    for _ in range(1, depth):
        mods.append(nn.GELU())
        mods.append(nn.Linear(hidden, hidden))
    return nn.Sequential(*mods)


class _LlagaKGEncoder(nn.Module):
    """Same recipe as the BGRL inductive-KG encoder (GCNConv stack with
    BatchNorm + PReLU), but the input goes through the LLaGA gft9-pretrained
    projector first, providing an LLM-aware feature initialization."""

    def __init__(self, hidden=768, projector_ckpt=None, n_gnn_layers=2):
        super().__init__()
        from torch_geometric.nn import GCNConv, BatchNorm

        self.projector = _build_projector(in_dim=768, hidden=4096, depth=2)
        if projector_ckpt:
            pt = torch.load(projector_ckpt, map_location="cpu", weights_only=False)
            self.projector.load_state_dict(pt, strict=True)
            print(f"[LLaGA] loaded projector from {projector_ckpt}")
        for p in self.projector.parameters():
            p.requires_grad = False

        self.compress = nn.Linear(4096, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        for _ in range(n_gnn_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.norms.append(BatchNorm(hidden))
            self.acts.append(nn.PReLU())

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor):
        with torch.no_grad():
            h = self.projector(x)
        h = self.compress(h)
        for conv, norm, act in zip(self.convs, self.norms, self.acts):
            h = conv(h, edge_index)
            h = norm(h)
            h = act(h)
        return h


_ARGS = None


def _builder(in_channels, device):
    model = _LlagaKGEncoder(
        hidden=_ARGS.hidden_dim,
        projector_ckpt=_ARGS.projector_ckpt,
        n_gnn_layers=_ARGS.n_gin_layers,
    )
    return model.to(device)


def main():
    global _ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--projector_ckpt",
                   default="ckpts/LLaGA/projector_gft9_vicuna.pt")
    p.add_argument("--hidden_dim", default=768, type=int)
    p.add_argument("--max_epochs", default=600, type=int)
    p.add_argument("--eval_every", default=20, type=int)
    p.add_argument("--patience", default=10, type=int)
    p.add_argument("--lr", default=2e-3, type=float)
    p.add_argument("--n_seeds", default=3, type=int)
    p.add_argument("--n_gin_layers", default=2, type=int)
    _ARGS = p.parse_args()

    # FB15K237 has 172K training edges; full-batch DistMult negatives OOM at
    # 5GB. Cap to 32K which keeps eval+train in 8GB (24GB GPU). Smaller graphs
    # (WN18RR 51K) won't be batched.
    pos_bs = 32768 if _ARGS.dataset.upper() in ("FB15K237", "FB15K-237") else None

    run_inductive_kg(
        method_tag="LLaGA",
        dataset=_ARGS.dataset,
        build_ft_model=_builder,
        hidden_dim=_ARGS.hidden_dim,
        lr=_ARGS.lr,
        weight_decay=1e-5,
        max_epochs=_ARGS.max_epochs,
        eval_every=_ARGS.eval_every,
        patience=_ARGS.patience,
        n_seeds=_ARGS.n_seeds,
        pos_batch_size=pos_bs,
    )


if __name__ == "__main__":
    main()
