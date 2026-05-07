"""GFT FT + structural (degree-based) fairness, encoder-only variant."""

import argparse
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = osp.dirname(osp.abspath(__file__))
_GFT_PARENT = osp.abspath(osp.join(_HERE, ".."))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", "..", ".."))
for p in (_HERE, _GFT_PARENT, _PROJECT_ROOT):
    if p not in sys.path: sys.path.insert(0, p)

from model.encoder import Encoder
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import compute_structural_fairness


class FTModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, dropout=0.2):
        super().__init__(); self.encoder = encoder; self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, edge_index, edge_attr=None):
        h = self.encoder(x, edge_index, edge_attr); h = self.dropout(h); return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool); m[idx] = True; return m


def _get_splits(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        return [(data.train_masks[i % avail].bool(),
                 data.val_masks[i % avail].bool(),
                 data.test_masks[i % avail].bool()) for i in range(5)]
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits; N = data.num_nodes
        tm = _idx_to_mask(s['train'], N); vm = _idx_to_mask(s.get('valid', s.get('val')), N); tsm = _idx_to_mask(s['test'], N)
        return [(tm, vm, tsm)] * 5
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        n = min(5, tm.size(1))
        return [(tm[:, i].bool(), vm[:, i].bool(),
                 (tsm[:, i] if tsm.dim() == 2 else tsm).bool()) for i in range(n)]
    return [(tm.bool(), vm.bool(), tsm.bool())] * 5


def train_ft(model, x, edge_index, edge_attr, y, tm, vm, lr, wd, max_epochs, patience):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(1, max_epochs + 1):
        model.train(); optim.zero_grad()
        F.cross_entropy(model(x, edge_index, edge_attr)[tm], y[tm]).backward(); optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr).argmax(-1)
            val_acc = (pred[vm] == y[vm]).float().mean().item() * 100.0
        if val_acc > best_val: best_val, best_state, no_imp = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(best_state); return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tolokers")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--backbone", default="sage")
    p.add_argument("--activation", default="relu")
    p.add_argument("--normalize", default="none")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--q", type=float, default=0.2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--num_splits", type=int, default=5)
    p.add_argument("--num_seeds", type=int, default=5)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GFT FT-STRUCT] dataset={args.dataset}")

    data, _ = load_tag(args.dataset)
    if data.x is None: data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat
    if data.y.dim() > 1: data.y = data.y.squeeze()
    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    input_dim = data.x.size(1)
    N = int(data.num_nodes)

    edge_attr = data.edge_text_feat.expand(data.edge_index.size(1), -1).contiguous()
    degree = torch.bincount(data.edge_index[0], minlength=N).cpu().numpy()
    print(f"[GFT FT-STRUCT] N={N} d={input_dim} C={num_classes} deg mean={degree.mean():.1f}")

    encoder_path = osp.join(args.ckpt_dir, "encoder.pt")
    state = torch.load(encoder_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    splits = _get_splits(data)
    SPLIT_SEEDS = [0, 1, 2, 3, 4]; RUN_SEEDS = [42, 43, 44, 45, 46]
    if args.debug:
        SPLIT_SEEDS = [0]; RUN_SEEDS = [42]
    else:
        n_split = min(args.num_splits, 5)
        n_seed = min(args.num_seeds, 5)
        SPLIT_SEEDS = SPLIT_SEEDS[:n_split]
        RUN_SEEDS = RUN_SEEDS[:n_seed]

    results = []
    for split_idx, (tm, vm, tsm) in enumerate(splits):
        if split_idx >= len(SPLIT_SEEDS): break
        tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
        for rs in RUN_SEEDS:
            torch.manual_seed(rs); np.random.seed(rs)
            activation_cls = nn.ReLU if args.activation == "relu" else nn.LeakyReLU
            encoder = Encoder(input_dim=input_dim, hidden_dim=args.hidden_dim,
                              activation=activation_cls, num_layers=args.num_layers,
                              backbone=args.backbone, normalize=args.normalize, dropout=args.dropout)
            res = encoder.load_state_dict(state, strict=False)
            if split_idx == 0 and rs == RUN_SEEDS[0]:
                print(f"  encoder ckpt: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
            model = FTModel(encoder, args.hidden_dim, num_classes, dropout=args.dropout).to(device)
            model = train_ft(model, data.x, data.edge_index, edge_attr, y, tm, vm,
                             args.lr, args.wd, args.max_epochs, args.patience)
            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index, edge_attr).argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            acc = (y_np[tsm_np] == preds[tsm_np]).mean() * 100.0
            struct = compute_structural_fairness(y_np, preds, degree, test_mask=tsm_np, q=args.q)
            print(f"[STRUCT_RAW] method=GFT dataset={args.dataset} split={split_idx} seed={rs} "
                  f"acc={acc:.4f} acc_head={struct['acc_head']:.4f} acc_tail={struct['acc_tail']:.4f} "
                  f"acc_gap={struct['acc_gap']:.4f} f1_head={struct['f1_head']:.4f} f1_tail={struct['f1_tail']:.4f} "
                  f"f1_gap={struct['f1_gap']:.4f} n_head={struct['n_head']} n_tail={struct['n_tail']} q={args.q}")
            results.append({'acc': acc, **struct})

    def _agg(k):
        vs = [r[k] for r in results
              if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs: return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))
    a, sa = _agg('acc'); ah, sah = _agg('acc_head'); at, sat = _agg('acc_tail'); ag, sag = _agg('acc_gap')
    fh, sfh = _agg('f1_head'); ft, sft = _agg('f1_tail'); fg, sfg = _agg('f1_gap')
    print(f'[STRUCT_AGG] method=GFT dataset={args.dataset} n_runs={len(results)} '
          f'acc="{a:.2f} ± {sa:.2f}" acc_head="{ah:.2f} ± {sah:.2f}" acc_tail="{at:.2f} ± {sat:.2f}" '
          f'acc_gap="{ag:.2f} ± {sag:.2f}" f1_head="{fh:.2f} ± {sfh:.2f}" f1_tail="{ft:.2f} ± {sft:.2f}" '
          f'f1_gap="{fg:.2f} ± {sfg:.2f}" q={args.q}')


if __name__ == "__main__":
    main()
