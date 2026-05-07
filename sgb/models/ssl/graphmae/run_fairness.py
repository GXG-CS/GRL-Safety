"""GraphMAE FT + demographic fairness on tolokers.

Clone of run_imbalance.py with:
  - No imbalance split (use clean train mask)
  - 5 split_seeds x 5 run_seeds = 25 runs
  - Load tolokers education binary as sensitive
  - Emit FAIR_RAW / FAIR_AGG lines with Delta_SP / Delta_EO / Delta_Utility
"""

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
from sklearn.metrics import f1_score, roc_auc_score

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from graphmae.models.edcoder import PreModel
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import load_tolokers_education_binary, compute_group_fairness


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
        h = self.pre_model.embed(g, x)
        h = self.dropout(h)
        return self.head(h)


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
    src, dst = raw_ei[0], raw_ei[1]
    base_ei = raw_ei[:, src != dst]
    g = dgl.graph((base_ei[0], base_ei[1]), num_nodes=N).remove_self_loop().add_self_loop().to(device)
    return g, feat.to(device), y.long().to(device), splits


def train_ft(model, g, feat, y, tm, vm, device, max_epochs, patience, lr, wd):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(1, max_epochs + 1):
        model.train(); optim.zero_grad()
        logits = model(g, feat)
        F.cross_entropy(logits[tm], y[tm]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(g, feat), dim=-1)[:, 1].cpu().numpy()
            yv = y[vm].cpu().numpy(); pv = probs[vm.cpu().numpy()]
            try: val_auc = roc_auc_score(yv, pv) * 100.0
            except Exception: val_auc = 0.0
        if val_auc > best_val:
            best_val = val_auc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tolokers")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--patience", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g, feat, y, splits = load_dataset(args.dataset, device)
    num_classes = int(y.max().item()) + 1
    N = feat.size(0)
    input_size = feat.size(1)

    sens, sens_meta = load_tolokers_education_binary()
    assert sens.numel() == N
    sens_np = sens.numpy()
    print(f"[GraphMAE FT-FAIR] N={N} d={input_size} C={num_classes}")
    print(f"[GraphMAE FT-FAIR] sens: {sens_meta}")

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    SPLIT_SEEDS = [0, 1, 2, 3, 4]; RUN_SEEDS = [42, 43, 44, 45, 46]
    if args.debug: SPLIT_SEEDS = [0]; RUN_SEEDS = [42]

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
                logits = model(g, feat)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            auc = roc_auc_score(y_np[tsm_np], probs[tsm_np]) * 100.0
            f1 = f1_score(y_np[tsm_np], preds[tsm_np], average='macro', zero_division=0) * 100.0
            fair = compute_group_fairness(y_np, preds, probs, sens_np, test_mask=tsm_np)
            print(f"[FAIR_RAW] method=GraphMAE_FT dataset={args.dataset} split={split_idx} seed={rs} "
                  f"auc={auc:.4f} macro_f1={f1:.4f} "
                  f"delta_sp={fair['delta_sp']:.4f} delta_eo={fair['delta_eo']:.4f} "
                  f"delta_utility={fair['delta_utility']:.4f} "
                  f"auc_s0={fair['auc_s0']} auc_s1={fair['auc_s1']} "
                  f"n_s0={fair['n_s0']} n_s1={fair['n_s1']}")
            results.append({'auc': auc, 'f1': f1, **fair})

    def _agg(k):
        vs = [r[k] for r in results if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs: return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))
    a, sa = _agg('auc'); f, sf = _agg('f1')
    sp, ssp = _agg('delta_sp'); eo, seo = _agg('delta_eo'); u, su = _agg('delta_utility')
    print(f'[FAIR_AGG] method=GraphMAE_FT dataset={args.dataset} n_runs={len(results)} '
          f'auc="{a:.2f} ± {sa:.2f}" macro_f1="{f:.2f} ± {sf:.2f}" '
          f'delta_sp="{sp:.4f} ± {ssp:.4f}" '
          f'delta_eo="{eo:.4f} ± {seo:.4f}" '
          f'delta_utility="{u:.4f} ± {su:.4f}"')


if __name__ == "__main__":
    main()
