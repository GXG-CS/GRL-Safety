"""GraphMAE inductive KG link-prediction on FB15K237/WN18RR."""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn
import dgl

_GMAE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_GMAE_DIR, "..", "..", ".."))
if _GMAE_DIR not in sys.path:
    sys.path.insert(0, _GMAE_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.ssl.graphmae.run_feature_noise import build_joint_model
from sgb.data.inductive_kg_runner import run_inductive_kg


class _KGModel(nn.Module):
    def __init__(self, pre_model):
        super().__init__()
        self.pre_model = pre_model

    def encode(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        g = dgl.graph((src, dst), num_nodes=x.size(0))
        g = g.remove_self_loop().add_self_loop().to(x.device)
        return self.pre_model.embed(g, x)


_CKPT = None
_ARGS = None


def _builder(in_channels, device):
    global _CKPT
    if _CKPT is None:
        ckpt = torch.load(_ARGS.ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        _CKPT = ckpt
    pre_model = build_joint_model(num_features=in_channels)
    pre_model.load_state_dict(_CKPT)
    return _KGModel(pre_model).to(device)


def main():
    global _ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="WN18RR")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--max_epochs", default=600, type=int)
    p.add_argument("--eval_every", default=20, type=int)
    p.add_argument("--patience", default=10, type=int)
    p.add_argument("--lr", default=2e-3, type=float)
    p.add_argument("--wd", default=1e-5, type=float)
    p.add_argument("--n_seeds", default=3, type=int)
    p.add_argument("--hidden_dim", default=768, type=int)
    _ARGS = p.parse_args()

    run_inductive_kg(
        method_tag="GraphMAE_FT", dataset=_ARGS.dataset,
        build_ft_model=_builder, hidden_dim=_ARGS.hidden_dim,
        lr=_ARGS.lr, weight_decay=_ARGS.wd,
        max_epochs=_ARGS.max_epochs, eval_every=_ARGS.eval_every,
        patience=_ARGS.patience, n_seeds=_ARGS.n_seeds,
    )


if __name__ == "__main__":
    main()
