"""OFA scaffold-OOD on BBBP / BACE.

OFA-lite encoder (`OFAEncoderHead.encoder` PyGRGCNEdge) loaded from pretrained
weights, mean-pool, linear head. We do not use OFA's prompt-graph + class-node
formulation (would require a separate pipeline); we benchmark OFA's *learned
representation* under a standard graph-cls FT, on equal footing with other
methods.
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
from sgb.data.scaffold_ood_runner import run_scaffold_ood


class _OFAGraphModel(nn.Module):
    def __init__(self, ofa_head, num_classes, dropout=0.2):
        super().__init__()
        self.ofa = ofa_head  # OFAEncoderHead — uses llm_proj + encoder + out_proj
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(ofa_head.emb_dim, num_classes)

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool

        # Build OFA-compatible graph: needs g.x, g.edge_index, g.edge_type, g.edge_attr.
        # Mol graphs in TAG cache only have g.x (768d node feat) and g.edge_index.
        # Synthesize zero edge_type and zero edge_attr to feed PyGRGCNEdge.
        x = batch.x  # [N_total, 768]
        edge_index = batch.edge_index
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=x.device)
        edge_attr = torch.zeros(edge_index.size(1), x.size(1), dtype=x.dtype, device=x.device)

        class _G:
            pass
        g = _G()
        g.x = x
        g.edge_index = edge_index
        g.edge_type = edge_type
        g.edge_attr = edge_attr

        # Use OFA's encoder pipeline manually (skip the class-text cosine head).
        x_proj = self.ofa.llm_proj(g.x)
        e_proj = self.ofa.llm_proj(g.edge_attr)
        g.x = x_proj
        g.edge_attr = e_proj
        node_emb = self.ofa.encoder(g)
        node_emb = self.ofa.out_proj(node_emb)

        h = global_mean_pool(node_emb, batch.batch)
        h = self.dropout(h)
        return self.head(h)


_CKPT_PATH = None


def _builder(in_channels, num_tasks, dropout, device):
    head = OFAEncoderHead(llm_name="ST", emb_dim=768, num_layers=6,
                          num_rels=5, dropout=0.0, jk="last")
    if _CKPT_PATH:
        payload = torch.load(_CKPT_PATH, map_location=device, weights_only=False)
        if "encoder" in payload:
            head.encoder.load_state_dict(payload["encoder"], strict=False)
        if "llm_proj" in payload:
            head.llm_proj.load_state_dict(payload["llm_proj"], strict=False)
    head = head.to(device)
    return _OFAGraphModel(head, num_classes=num_tasks, dropout=dropout).to(device)


def main():
    global _CKPT_PATH
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt_path",
                   default="sgb/models/gfm/ofa/ckpts/pretrain_e2e/encoder_weights_final.pt",
                   help="OFA encoder weights from load_pretrained.py")
    p.add_argument("--max_epochs", default=200, type=int)
    p.add_argument("--patience", default=50, type=int)
    p.add_argument("--lr", default=5e-4, type=float)
    p.add_argument("--wd", default=1e-5, type=float)
    p.add_argument("--dropout", default=0.2, type=float)
    p.add_argument("--n_seeds", default=5, type=int)
    args = p.parse_args()

    _CKPT_PATH = args.ckpt_path

    run_scaffold_ood(
        method_tag="OFA",
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
