"""UniGraph2 FT + step imbalance evaluation (node classification only)."""

import copy
import os
import os.path as osp
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import balanced_accuracy_score

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.unigraph2 import UniGraph2  # type: ignore
from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance, compute_imbalance_metrics


def build_model(num_features=768, num_hidden=768, num_layers=3,
                num_experts=8, num_selected_experts=2,
                feat_drop_rate=0.1, edge_mask_rate=0.1,
                gamma=2.0, lambda_spd=0.5):
    return UniGraph2(
        input_dims={"text": num_features},
        hidden_dim=num_hidden,
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        num_layers=num_layers,
        feat_drop_rate=feat_drop_rate,
        edge_mask_rate=edge_mask_rate,
        gamma=gamma,
        lambda_spd=lambda_spd,
    )


class FTModel(nn.Module):
    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)

    def forward(self, g, x):
        h = self.pre_model(g, {"text": x}, spd_matrix=None, return_embeddings=True)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def load_dataset(name, device):
    data, _ = load_tag(name)
    if data.x is not None and data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat[data.x].float()
    elif data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        feat = data.x.float()
    elif hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat.float()
    else:
        raise RuntimeError(f"Cannot extract 768d for {name}")

    y = data.y.squeeze() if data.y is not None and data.y.dim() > 1 else data.y
    src, dst = data.edge_index[0], data.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=feat.size(0))
    g = g.remove_self_loop().add_self_loop()

    N = feat.size(0)
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        tm, vm, tsm = data.train_masks[0].bool(), data.val_masks[0].bool(), data.test_masks[0].bool()
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
    else:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            tm, vm, tsm = tm[:, 0].bool(), vm[:, 0].bool(), (tsm[:, 0] if tsm.dim() == 2 else tsm).bool()
        else:
            tm, vm, tsm = tm.bool(), vm.bool(), tsm.bool()

    return g.to(device), feat.to(device), y.long().to(device), tm, vm, tsm


def train_ft(model, g, feat, y, train_mask, val_mask, device,
             max_epochs, patience, lr, wd):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(g, feat)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(g, feat).argmax(-1)
            val_bacc = balanced_accuracy_score(
                y[val_mask].cpu().numpy(), pred[val_mask].cpu().numpy()
            ) * 100.0
        if val_bacc > best_val:
            best_val = val_bacc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--rho", type=int, default=10)
    parser.add_argument("--n_reps", type=int, default=10)
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--patience", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[UG2 FT-IMB] device={device}, dataset={args.dataset}, rho={args.rho}, n_reps={args.n_reps}")

    g, feat, y, train_mask_base, val_mask, test_mask = load_dataset(args.dataset, device)
    num_classes = int(y.max().item()) + 1
    input_size = feat.size(1)

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    print(f"[UG2 FT-IMB] N={g.num_nodes()}, d={input_size}, C={num_classes}")

    all_results = []
    for rep_idx in range(args.n_reps):
        seed = rep_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        imb_train_mask, meta = make_step_imbalance(
            train_mask=train_mask_base, y=y, rho=args.rho, seed=seed)
        imb_train_mask = imb_train_mask.to(device)

        if rep_idx == 0:
            print(f"  [rep 0 meta] n_major_max={meta['n_major_max']} "
                  f"n_minor_target={meta['n_minor_target']} "
                  f"minor_classes={meta['minor_classes']}")

        pre_model = build_model(num_features=input_size)
        pre_model.load_state_dict(state)
        model = FTModel(pre_model, num_hidden=768, num_classes=num_classes,
                        dropout=args.dropout).to(device)
        model = train_ft(model, g, feat, y, imb_train_mask, val_mask, device,
                         args.max_epochs, args.patience, args.lr, args.wd)

        model.eval()
        with torch.no_grad():
            pred = model(g, feat).argmax(-1)
        metrics = compute_imbalance_metrics(y[test_mask].cpu().numpy(),
                                            pred[test_mask].cpu().numpy(),
                                            num_classes=num_classes)
        all_results.append({"rep": rep_idx, **metrics})

        print(f"[IMB_RAW] method=UniGraph2_FT dataset={args.dataset} rho={args.rho} "
              f"rep={rep_idx} seed={seed} "
              f"bacc={metrics['bacc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"acc={metrics['acc']:.4f} n_minor_target={meta['n_minor_target']}")
        print(f"[IMB_PER_CLASS] method=UniGraph2_FT dataset={args.dataset} rho={args.rho} "
              f"rep={rep_idx} minor_classes={meta['minor_classes']} "
              f"per_class_recall={metrics['per_class_recall']} "
              f"per_class_f1={metrics['per_class_f1']}")

    bacc = np.array([r['bacc'] for r in all_results])
    f1 = np.array([r['macro_f1'] for r in all_results])
    acc = np.array([r['acc'] for r in all_results])
    print(f"\n=== UG2 Imbalance ({args.dataset}, rho={args.rho}, n={len(all_results)}) ===")
    print(f"  bAcc={bacc.mean():.2f} ± {bacc.std():.2f}  macroF1={f1.mean():.2f} ± {f1.std():.2f}")
    print(f"[IMB_AGG] method=UniGraph2_FT dataset={args.dataset} rho={args.rho} "
          f"n_reps={len(all_results)} "
          f"bacc=\"{bacc.mean():.2f} ± {bacc.std():.2f}\" "
          f"macro_f1=\"{f1.mean():.2f} ± {f1.std():.2f}\" "
          f"acc=\"{acc.mean():.2f} ± {acc.std():.2f}\"")


if __name__ == "__main__":
    main()
