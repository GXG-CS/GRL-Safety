"""GraphMAE FT + structural (degree-based) fairness."""

import argparse
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from graphmae.models.edcoder import PreModel
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import compute_structural_fairness


def build_joint_model(num_features=768, num_hidden=768, num_layers=2, num_heads=4):
    return PreModel(
        in_dim=num_features, num_hidden=num_hidden, num_layers=num_layers,
        nhead=num_heads, nhead_out=1, activation="prelu",
        feat_drop=0.2, attn_drop=0.1, negative_slope=0.2,
        residual=False, norm=None, mask_rate=0.5,
        encoder_type="gat", decoder_type="mlp", loss_fn="sce",
        drop_edge_rate=0.0, replace_rate=0.0, alpha_l=3.0, concat_hidden=False,
    )


class FTModel(nn.Module):
    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)
    def forward(self, g, x):
        h = self.pre_model.embed(g, x); h = self.dropout(h); return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool); m[idx] = True; return m


def load_dataset(name, device):
    data, _ = load_tag(name)
    if data.x is not None and data.x.dtype == torch.long and hasattr(data, "node_text_feat"):
        feat = data.node_text_feat[data.x].float()
    elif data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        feat = data.x.float()
    elif hasattr(data, "node_text_feat"):
        feat = data.node_text_feat.float()
    else:
        raise RuntimeError("no 768d features")
    y = data.y.squeeze() if data.y is not None and data.y.dim() > 1 else data.y
    N = int(data.num_nodes)
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        splits = [(data.train_masks[i % avail].bool(), data.val_masks[i % avail].bool(),
                   data.test_masks[i % avail].bool()) for i in range(5)]
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        tm = _idx_to_mask(s['train'], N); vm = _idx_to_mask(s.get('valid', s.get('val')), N); tsm = _idx_to_mask(s['test'], N)
        splits = [(tm, vm, tsm)] * 5
    else:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            splits = [(tm[:, i].bool(), vm[:, i].bool(),
                       (tsm[:, i] if tsm.dim() == 2 else tsm).bool()) for i in range(min(5, tm.size(1)))]
        else:
            splits = [(tm.bool(), vm.bool(), tsm.bool())] * 5
    raw_ei = data.edge_index.long()
    degree = torch.bincount(raw_ei[0], minlength=N).numpy()
    src, dst = raw_ei[0], raw_ei[1]
    base_ei = raw_ei[:, src != dst]
    g = dgl.graph((base_ei[0], base_ei[1]), num_nodes=N).remove_self_loop().add_self_loop().to(device)
    return g, feat.to(device), y.long().to(device), splits, degree


def train_ft(model, g, feat, y, tm, vm, device, max_epochs, patience, lr, wd):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(1, max_epochs + 1):
        model.train(); optim.zero_grad()
        F.cross_entropy(model(g, feat)[tm], y[tm]).backward(); optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(g, feat).argmax(-1)
            val_acc = (pred[vm] == y[vm]).float().mean().item() * 100.0
        if val_acc > best_val: best_val = val_acc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(best_state); return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tolokers")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--max_epochs", default=500, type=int)
    p.add_argument("--patience", default=200, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.2, type=float)
    p.add_argument("--q", default=0.2, type=float)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--num_splits", type=int, default=5)
    p.add_argument("--num_seeds", type=int, default=5)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g, feat, y, splits, degree = load_dataset(args.dataset, device)
    num_classes = int(y.max().item()) + 1
    N = feat.size(0); input_size = feat.size(1)
    print(f"[GraphMAE FT-STRUCT] N={N} d={input_size} C={num_classes} deg mean={degree.mean():.1f}")

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

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
            pre_model = build_joint_model(num_features=input_size)
            pre_model.load_state_dict(state)
            model = FTModel(pre_model, 768, num_classes, args.dropout).to(device)
            model = train_ft(model, g, feat, y, tm, vm, device,
                             args.max_epochs, args.patience, args.lr, args.wd)
            model.eval()
            with torch.no_grad():
                preds = model(g, feat).argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            acc = (y_np[tsm_np] == preds[tsm_np]).mean() * 100.0
            struct = compute_structural_fairness(y_np, preds, degree, test_mask=tsm_np, q=args.q)
            print(f"[STRUCT_RAW] method=GraphMAE_FT dataset={args.dataset} split={split_idx} seed={rs} "
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
    print(f'[STRUCT_AGG] method=GraphMAE_FT dataset={args.dataset} n_runs={len(results)} '
          f'acc="{a:.2f} ± {sa:.2f}" acc_head="{ah:.2f} ± {sah:.2f}" acc_tail="{at:.2f} ± {sat:.2f}" '
          f'acc_gap="{ag:.2f} ± {sag:.2f}" f1_head="{fh:.2f} ± {sfh:.2f}" f1_tail="{ft:.2f} ± {sft:.2f}" '
          f'f1_gap="{fg:.2f} ± {sfg:.2f}" q={args.q}')


if __name__ == "__main__":
    main()
