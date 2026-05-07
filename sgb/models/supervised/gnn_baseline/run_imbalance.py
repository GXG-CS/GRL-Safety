"""GNN baseline (GCN/GAT/SAGE) FT + step imbalance evaluation.

Step imbalance protocol follows TAM (ICML 2022) / ReNode (NeurIPS 2021):
  - minor = |Y|/2 lowest-count classes
  - n_minor = max(major class count) / rho
  - train set modified; val/test unchanged

Runs N_REPS repetitions per (dataset, rho); reports bAcc + macro-F1 +
per-class recall. Uniform FT hyperparameters (same as other safety dims).
"""

import copy
import os
import os.path as osp
import sys
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.supervised.gnn_baseline import GNNEncoderWrapper, METHOD_NAMES
from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance, compute_imbalance_metrics


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG dataset name (node-classif only).')
flags.DEFINE_string('model', 'gcn', 'GNN baseline: gcn, gat, or sage.')
flags.DEFINE_integer('rho', 10, 'Imbalance ratio (n_major_max / n_minor).')
flags.DEFINE_integer('n_reps', 10, 'Number of random repetitions (seeds).')
flags.DEFINE_integer('hidden', 768, 'Hidden dim.')
flags.DEFINE_integer('num_layers', 2, 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs per rep.')
flags.DEFINE_integer('patience', 200, 'Early stop patience (by val bAcc).')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')


class FTModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        h = self.encoder(data)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_base_masks(data):
    """Return (train, val, test) bool masks from data, taking split_idx=0."""
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


def train_ft(model, data, y, train_mask, val_mask, device):
    """Train by val bAcc early-stop."""
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    best_val_bacc, best_state, no_improve = -1.0, None, 0

    from sklearn.metrics import balanced_accuracy_score

    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(-1)
            val_pred = pred[val_mask].cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_bacc = balanced_accuracy_score(val_true, val_pred) * 100.0

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FLAGS.patience:
                break

    model.load_state_dict(best_state)
    return model


def _build_encoder(in_channels):
    return GNNEncoderWrapper(
        model_name=FLAGS.model,
        in_channels=in_channels,
        hidden_channels=FLAGS.hidden,
        num_layers=FLAGS.num_layers,
        dropout=FLAGS.dropout,
    )


def method_tag():
    return METHOD_NAMES[FLAGS.model.lower()]


def main(argv):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tag = method_tag()
    print(f"[{tag} FT-IMB] Using {device}, dataset={FLAGS.dataset}, rho={FLAGS.rho}, n_reps={FLAGS.n_reps}")

    data, _ = load_tag(FLAGS.dataset)
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
    input_size = data.x.size(1)

    train_mask_base, val_mask, test_mask = _get_base_masks(data)
    train_mask_base = train_mask_base.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"[{tag} FT-IMB] N={data.num_nodes}, d={input_size}, C={num_classes}")
    print(f"  base train={int(train_mask_base.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}")

    all_results = []

    for rep_idx in range(FLAGS.n_reps):
        seed = rep_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Apply step imbalance to train only
        imb_train_mask, meta = make_step_imbalance(
            train_mask=train_mask_base,
            y=y,
            rho=FLAGS.rho,
            seed=seed,
        )
        imb_train_mask = imb_train_mask.to(device)

        if rep_idx == 0:
            print(f"  [rep {rep_idx}] meta: n_major_max={meta['n_major_max']} "
                  f"n_minor_target={meta['n_minor_target']} "
                  f"minor_classes={meta['minor_classes']}")
            print(f"  [rep {rep_idx}] counts_before={meta['counts_before']} "
                  f"counts_after={meta['counts_after']}")

        # Fresh encoder per rep
        encoder = _build_encoder(input_size)
        model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)
        model = train_ft(model, data, y, imb_train_mask, val_mask, device)

        # Evaluate on test
        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(-1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        metrics = compute_imbalance_metrics(y_true, y_pred, num_classes=num_classes)

        all_results.append({"rep": rep_idx, **metrics})

        print(f"[IMB_RAW] method={tag} dataset={FLAGS.dataset} rho={FLAGS.rho} "
              f"rep={rep_idx} seed={seed} "
              f"bacc={metrics['bacc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"acc={metrics['acc']:.4f} "
              f"n_minor_target={meta['n_minor_target']}")
        print(f"[IMB_PER_CLASS] method={tag} dataset={FLAGS.dataset} rho={FLAGS.rho} "
              f"rep={rep_idx} minor_classes={meta['minor_classes']} "
              f"per_class_recall={metrics['per_class_recall']} "
              f"per_class_f1={metrics['per_class_f1']}")

    # Aggregate
    bacc = np.array([r['bacc'] for r in all_results])
    f1 = np.array([r['macro_f1'] for r in all_results])
    acc = np.array([r['acc'] for r in all_results])
    print(f"\n=== {tag} Imbalance Results ({FLAGS.dataset}, rho={FLAGS.rho}, n={len(all_results)}) ===")
    print(f"  bAcc    = {bacc.mean():.2f} ± {bacc.std():.2f}")
    print(f"  macroF1 = {f1.mean():.2f} ± {f1.std():.2f}")
    print(f"  acc     = {acc.mean():.2f} ± {acc.std():.2f}")
    print(f"[IMB_AGG] method={tag} dataset={FLAGS.dataset} rho={FLAGS.rho} "
          f"n_reps={len(all_results)} "
          f"bacc=\"{bacc.mean():.2f} ± {bacc.std():.2f}\" "
          f"macro_f1=\"{f1.mean():.2f} ± {f1.std():.2f}\" "
          f"acc=\"{acc.mean():.2f} ± {acc.std():.2f}\"")


if __name__ == "__main__":
    app.run(main)
