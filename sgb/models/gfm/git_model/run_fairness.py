"""GIT FT + demographic fairness on tolokers.

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
from sklearn.metrics import f1_score, roc_auc_score

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
for p in (_HERE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.encoder import Encoder
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import load_tolokers_education_binary, compute_group_fairness


class FTModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_splits(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        return [(data.train_masks[i % avail].bool(),
                 data.val_masks[i % avail].bool(),
                 data.test_masks[i % avail].bool()) for i in range(5)]
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        N = data.num_nodes
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
        return [(tm, vm, tsm)] * 5
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        n = min(5, tm.size(1))
        return [(tm[:, i].bool(), vm[:, i].bool(),
                 (tsm[:, i] if tsm.dim() == 2 else tsm).bool()) for i in range(n)]
    return [(tm.bool(), vm.bool(), tsm.bool())] * 5


def load_encoder_ckpt(encoder, ckpt_dir, ckpt_name="encoder.pt"):
    path = osp.join(ckpt_dir, ckpt_name)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    res = encoder.load_state_dict(state, strict=False)
    return encoder, res


def train_ft(model, x, edge_index, y, tm, vm, lr, wd, max_epochs, patience):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = -1.0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index)
        F.cross_entropy(logits[tm], y[tm]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x, edge_index), dim=-1)[:, 1].cpu().numpy()
            yv = y[vm].cpu().numpy()
            pv = probs[vm.cpu().numpy()]
            try:
                val_auc = roc_auc_score(yv, pv) * 100.0
            except Exception:
                val_auc = 0.0
        if val_auc > best_val:
            best_val, best_state, no_imp = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tolokers")
    p.add_argument("--ckpt_dir", required=True, help="Directory containing encoder.pt")
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--backbone", type=str, default="sage")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--normalize", type=str, default="none")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GIT FT-FAIR] device={device}, dataset={args.dataset}")

    data, _ = load_tag(args.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat
    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    input_dim = data.x.size(1)
    N = int(data.num_nodes)

    sens, sens_meta = load_tolokers_education_binary()
    assert sens.numel() == N, f"sens {sens.numel()} vs N {N}"
    sens_np = sens.numpy()
    print(f"[GIT FT-FAIR] N={N}, d={input_dim}, C={num_classes}")
    print(f"[GIT FT-FAIR] sens: {sens_meta}")

    splits = _get_splits(data)
    SPLIT_SEEDS = [0, 1, 2, 3, 4]
    RUN_SEEDS = [42, 43, 44, 45, 46]
    if args.debug:
        SPLIT_SEEDS = [0]
        RUN_SEEDS = [42]

    results = []
    for split_idx, (tm, vm, tsm) in enumerate(splits):
        if split_idx >= len(SPLIT_SEEDS):
            break
        tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
        for rs in RUN_SEEDS:
            torch.manual_seed(rs); np.random.seed(rs)
            activation_cls = nn.ReLU if args.activation == "relu" else nn.LeakyReLU
            encoder = Encoder(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                activation=activation_cls,
                num_layers=args.num_layers,
                backbone=args.backbone,
                normalize=args.normalize,
                dropout=args.dropout,
            )
            encoder, res = load_encoder_ckpt(encoder, args.ckpt_dir)
            if split_idx == 0 and rs == RUN_SEEDS[0]:
                print(f"  encoder ckpt: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

            model = FTModel(encoder, args.hidden_dim, num_classes, dropout=args.dropout).to(device)
            model = train_ft(model, data.x, data.edge_index, y, tm, vm,
                             args.lr, args.wd, args.max_epochs, args.patience)

            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            auc = roc_auc_score(y_np[tsm_np], probs[tsm_np]) * 100.0
            f1 = f1_score(y_np[tsm_np], preds[tsm_np], average='macro', zero_division=0) * 100.0
            fair = compute_group_fairness(y_np, preds, probs, sens_np, test_mask=tsm_np)
            print(f"[FAIR_RAW] method=GIT dataset={args.dataset} split={split_idx} seed={rs} "
                  f"auc={auc:.4f} macro_f1={f1:.4f} "
                  f"delta_sp={fair['delta_sp']:.4f} delta_eo={fair['delta_eo']:.4f} "
                  f"delta_utility={fair['delta_utility']:.4f} "
                  f"auc_s0={fair['auc_s0']} auc_s1={fair['auc_s1']} "
                  f"n_s0={fair['n_s0']} n_s1={fair['n_s1']}")
            results.append({'auc': auc, 'f1': f1, **fair})

    def _agg(k):
        vs = [r[k] for r in results
              if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs:
            return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))
    a, sa = _agg('auc'); f, sf = _agg('f1')
    sp, ssp = _agg('delta_sp'); eo, seo = _agg('delta_eo'); u, su = _agg('delta_utility')
    print(f'[FAIR_AGG] method=GIT dataset={args.dataset} n_runs={len(results)} '
          f'auc="{a:.2f} ± {sa:.2f}" macro_f1="{f:.2f} ± {sf:.2f}" '
          f'delta_sp="{sp:.4f} ± {ssp:.4f}" '
          f'delta_eo="{eo:.4f} ± {seo:.4f}" '
          f'delta_utility="{u:.4f} ± {su:.4f}"')


if __name__ == "__main__":
    main()
