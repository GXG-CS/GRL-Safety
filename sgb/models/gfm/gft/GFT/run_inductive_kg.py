"""GFT inductive KG link-prediction on FB15K237/WN18RR.

Loads GFT's pretrained `Encoder` (SAGE/2-layer/768d, batch-norm) and exposes
`.encode(x, edge_index)` for the shared inductive_kg runner. Codebook/proto
not used: encoder + DistMult decoder, mirroring the other 8 methods.

The MySAGEConv `message(x_j, xe)` was patched to handle `xe=None` (KG TAGs
do not carry edge_attr through this code path).
"""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", "..", ".."))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.encoder import Encoder
from sgb.data.inductive_kg_runner import run_inductive_kg


class _KGModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder.encode(x, edge_index)


_ARGS = None


def _builder(in_channels, device):
    encoder = Encoder(
        input_dim=in_channels,
        hidden_dim=_ARGS.hidden_dim,
        activation=nn.ReLU,
        num_layers=_ARGS.num_layers,
        backbone=_ARGS.backbone,
        normalize="batch",
        dropout=0.0,
    )
    if _ARGS.ckpt_path:
        sd = torch.load(_ARGS.ckpt_path, map_location="cpu", weights_only=False)
        miss, unexp = encoder.load_state_dict(sd, strict=False)
        if miss or unexp:
            print(f"[GFT] encoder load: missing={len(miss)} unexpected={len(unexp)}")
    return _KGModel(encoder).to(device)


def main():
    global _ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="WN18RR")
    p.add_argument("--ckpt_path",
                   default=osp.join(_PROJECT_ROOT, "ckpts", "GFT", "encoder.pt"))
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--backbone", default="sage")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--max_epochs", type=int, default=600)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n_seeds", type=int, default=3)
    _ARGS = p.parse_args()

    run_inductive_kg(
        method_tag="GFT",
        dataset=_ARGS.dataset,
        build_ft_model=_builder,
        hidden_dim=_ARGS.hidden_dim,
        lr=_ARGS.lr,
        weight_decay=_ARGS.wd,
        max_epochs=_ARGS.max_epochs,
        eval_every=_ARGS.eval_every,
        patience=_ARGS.patience,
        n_seeds=_ARGS.n_seeds,
    )


if __name__ == "__main__":
    main()
