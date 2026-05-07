"""OFA inductive KG link-prediction on FB15K237 / WN18RR.

OFA-lite encoder (`OFAEncoderHead.encoder` PyGRGCNEdge + llm_proj) loaded
from pretrained weights, exposing `.encode(x, edge_index) -> z [N, d]` for
the shared inductive-KG runner. Edge_type / edge_attr synthesized as zeros
since the inductive-KG runner builds compact-id subgraphs without OFA's
prompt-graph relation slots.
"""
import argparse
import os.path as osp
import sys

import torch
import torch.nn as nn

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.gfm.ofa.ofa_encoder_head import OFAEncoderHead
from sgb.data.inductive_kg_runner import run_inductive_kg


class _OFAKGModel(nn.Module):
    def __init__(self, ofa_head):
        super().__init__()
        self.ofa = ofa_head

    def encode(self, x, edge_index):
        device = x.device
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=device)
        edge_attr = torch.zeros(edge_index.size(1), x.size(1), dtype=x.dtype, device=device)

        x_proj = self.ofa.llm_proj(x)
        e_proj = self.ofa.llm_proj(edge_attr)

        class _G:
            pass
        g = _G()
        g.x = x_proj
        g.edge_index = edge_index
        g.edge_type = edge_type
        g.edge_attr = e_proj

        z = self.ofa.encoder(g)
        z = self.ofa.out_proj(z)
        return z


_CKPT_PATH = None


def _builder(in_channels, device):
    head = OFAEncoderHead(llm_name="ST", emb_dim=768, num_layers=6,
                          num_rels=5, dropout=0.0, jk="last")
    if _CKPT_PATH:
        payload = torch.load(_CKPT_PATH, map_location=device, weights_only=False)
        if "encoder" in payload:
            head.encoder.load_state_dict(payload["encoder"], strict=False)
        if "llm_proj" in payload:
            head.llm_proj.load_state_dict(payload["llm_proj"], strict=False)
    head = head.to(device)
    return _OFAKGModel(head).to(device)


def main():
    global _CKPT_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="WN18RR")
    p.add_argument("--ckpt_path",
                   default="sgb/models/gfm/ofa/ckpts/pretrain_e2e/encoder_weights_final.pt")
    p.add_argument("--max_epochs", default=600, type=int)
    p.add_argument("--eval_every", default=20, type=int)
    p.add_argument("--patience", default=10, type=int)
    p.add_argument("--lr", default=2e-3, type=float)
    p.add_argument("--wd", default=1e-5, type=float)
    p.add_argument("--n_seeds", default=3, type=int)
    p.add_argument("--hidden_dim", default=768, type=int)
    p.add_argument("--pos_batch_size", default=None, type=int)
    args = p.parse_args()

    _CKPT_PATH = args.ckpt_path

    run_inductive_kg(
        method_tag="OFA_FT", dataset=args.dataset,
        build_ft_model=_builder, hidden_dim=args.hidden_dim,
        lr=args.lr, weight_decay=args.wd,
        max_epochs=args.max_epochs, eval_every=args.eval_every,
        patience=args.patience, n_seeds=args.n_seeds,
        pos_batch_size=args.pos_batch_size,
    )


if __name__ == "__main__":
    main()
