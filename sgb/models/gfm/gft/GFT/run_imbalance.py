"""GFT FT + step imbalance (node classification, encoder-only variant).

Note: GFT's native FT uses encoder + VQ prototype classifier. For imbalance
experiments, we use the encoder with a simple linear head. Per user notes,
proto vs no_proto difference is <1pt, so this is a valid simplification.
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
from sklearn.metrics import balanced_accuracy_score

_HERE = osp.dirname(osp.abspath(__file__))
_GFT_PARENT = osp.abspath(osp.join(_HERE, ".."))  # .../gft/
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", "..", ".."))
for p in (_HERE, _GFT_PARENT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.encoder import Encoder
from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance, compute_imbalance_metrics


class FTModel(nn.Module):
    def __init__(self, encoder, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.encoder(x, edge_index, edge_attr)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_base_masks(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        return data.train_masks[0].bool(), data.val_masks[0].bool(), data.test_masks[0].bool()
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        N = data.num_nodes
        return (_idx_to_mask(s['train'], N),
                _idx_to_mask(s.get('valid', s.get('val')), N),
                _idx_to_mask(s['test'], N))
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        return tm[:, 0].bool(), vm[:, 0].bool(), (tsm[:, 0] if tsm.dim() == 2 else tsm).bool()
    return tm.bool(), vm.bool(), tsm.bool()


def train_ft(model, x, edge_index, edge_attr, y, tm, vm, lr, wd, max_epochs, patience):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_imp = -1.0, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index, edge_attr)
        F.cross_entropy(logits[tm], y[tm]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index, edge_attr).argmax(-1)
            val_bacc = balanced_accuracy_score(y[vm].cpu().numpy(), pred[vm].cpu().numpy()) * 100.0
        if val_bacc > best_val:
            best_val, best_state, no_imp = val_bacc, copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--ckpt_dir", required=True, help="Directory containing encoder.pt")
    p.add_argument("--rho", type=int, default=10)
    p.add_argument("--n_reps", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--backbone", type=str, default="sage")
    p.add_argument("--activation", type=str, default="relu")
    p.add_argument("--normalize", type=str, default="none")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=200)
    args = p.parse_args()

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GFT FT-IMB] device={device}, dataset={args.dataset}, rho={args.rho}, n_reps={args.n_reps}")

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

    # GFT encoder requires per-edge edge_attr. Our TAG datasets have
    # edge_text_feat of shape [1, 768]; expand to [E, 768].
    edge_attr = data.edge_text_feat.expand(data.edge_index.size(1), -1).contiguous()

    train_mask_base, val_mask, test_mask = _get_base_masks(data)
    train_mask_base = train_mask_base.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"[GFT FT-IMB] N={data.num_nodes}, d={input_dim}, C={num_classes}")

    # Load encoder state
    encoder_path = osp.join(args.ckpt_dir, "encoder.pt")
    state = torch.load(encoder_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

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
        res = encoder.load_state_dict(state, strict=False)
        if rep_idx == 0:
            print(f"  encoder ckpt: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")

        model = FTModel(encoder, args.hidden_dim, num_classes, dropout=args.dropout).to(device)
        model = train_ft(model, data.x, data.edge_index, edge_attr, y, imb_train_mask, val_mask,
                         args.lr, args.wd, args.max_epochs, args.patience)

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index, edge_attr).argmax(-1)
        metrics = compute_imbalance_metrics(y[test_mask].cpu().numpy(),
                                            pred[test_mask].cpu().numpy(),
                                            num_classes=num_classes)
        all_results.append({"rep": rep_idx, **metrics})

        print(f"[IMB_RAW] method=GFT dataset={args.dataset} rho={args.rho} "
              f"rep={rep_idx} seed={seed} "
              f"bacc={metrics['bacc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"acc={metrics['acc']:.4f} n_minor_target={meta['n_minor_target']}")
        print(f"[IMB_PER_CLASS] method=GFT dataset={args.dataset} rho={args.rho} "
              f"rep={rep_idx} minor_classes={meta['minor_classes']} "
              f"per_class_recall={metrics['per_class_recall']} "
              f"per_class_f1={metrics['per_class_f1']}")

    bacc = np.array([r['bacc'] for r in all_results])
    f1 = np.array([r['macro_f1'] for r in all_results])
    acc = np.array([r['acc'] for r in all_results])
    print(f"\n=== GFT Imbalance ({args.dataset}, rho={args.rho}) ===")
    print(f"  bAcc={bacc.mean():.2f} ± {bacc.std():.2f}  macroF1={f1.mean():.2f} ± {f1.std():.2f}")
    print(f"[IMB_AGG] method=GFT dataset={args.dataset} rho={args.rho} "
          f"n_reps={len(all_results)} "
          f"bacc=\"{bacc.mean():.2f} ± {bacc.std():.2f}\" "
          f"macro_f1=\"{f1.mean():.2f} ± {f1.std():.2f}\" "
          f"acc=\"{acc.mean():.2f} ± {acc.std():.2f}\"")


if __name__ == "__main__":
    main()
