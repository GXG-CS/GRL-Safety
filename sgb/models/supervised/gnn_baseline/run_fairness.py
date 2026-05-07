"""GNN baseline (GCN/GAT/SAGE) FT + demographic fairness on tolokers.

Protocol:
  - Train on standard tolokers split (banned binary), no perturbation.
  - On test set, compute:
      * AUC-ROC (primary, matches original tolokers metric)
      * Per-group AUC for education-binary (high vs not-high)
      * Delta_SP / Delta_EO / Delta_Utility (PyGDebias standard)
  - 5 split_seeds x 5 run_seeds = 25 runs.
"""

import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from sklearn.metrics import f1_score, roc_auc_score

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.supervised.gnn_baseline import GNNEncoderWrapper, METHOD_NAMES
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import load_tolokers_education_binary, compute_group_fairness


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'tolokers', 'Fixed to tolokers for v1.')
flags.DEFINE_string('model', 'gcn', 'GNN baseline: gcn, gat, or sage.')
flags.DEFINE_integer('hidden', 768, 'Hidden dim.')
flags.DEFINE_integer('num_layers', 2, 'Number of encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')
flags.DEFINE_bool('debug', False, 'smoke: 1 split 1 run.')

SPLIT_SEEDS = [0, 1, 2, 3, 4]
RUN_SEEDS = [42, 43, 44, 45, 46]


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


def _build_splits(data):
    """tolokers has splits dict (train/val/test idx)."""
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        n = min(5, len(data.train_masks))
        return [(data.train_masks[i].bool(), data.val_masks[i].bool(),
                 data.test_masks[i].bool()) for i in range(n)]
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


def train_ft(model, data, y, train_mask, val_mask, test_mask, device):
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr,
                              weight_decay=FLAGS.weight_decay)
    best_val, best_state, no_imp = -1.0, None, 0
    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            logits_v = model(data)
            p = torch.softmax(logits_v, dim=-1)[:, 1].cpu().numpy()
            yv = y[val_mask].cpu().numpy()
            pv = p[val_mask.cpu().numpy()]
            try:
                val_auc = roc_auc_score(yv, pv) * 100.0
            except Exception:
                val_auc = 0.0
        if val_auc > best_val:
            best_val = val_auc
            best_state = copy.deepcopy(model.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= FLAGS.patience:
                break
    model.load_state_dict(best_state)
    return model


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = METHOD_NAMES.get(FLAGS.model, FLAGS.model.upper())
    print(f"[{tag} FT-FAIR] Using {device}, dataset={FLAGS.dataset}")

    # Load graph data
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
    N = int(data.num_nodes)
    input_size = data.x.size(1)
    print(f"[{tag} FT-FAIR] N={N}, d={input_size}, C={num_classes}")

    # Load sensitive attribute
    sens, sens_meta = load_tolokers_education_binary()
    if sens.numel() != N:
        raise RuntimeError(
            f"sensitive length {sens.numel()} != num_nodes {N}")
    sens_np = sens.numpy()
    print(f"[{tag} FT-FAIR] sensitive: {sens_meta}")

    splits = _build_splits(data)
    split_seeds = SPLIT_SEEDS[:1] if FLAGS.debug else SPLIT_SEEDS
    run_seeds = RUN_SEEDS[:1] if FLAGS.debug else RUN_SEEDS

    results = []
    for split_idx, (tm, vm, tsm) in enumerate(splits):
        if split_idx >= len(split_seeds): break
        tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
        for run_seed in run_seeds:
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            enc = GNNEncoderWrapper(
                FLAGS.model, in_channels=input_size,
                hidden_channels=FLAGS.hidden, num_layers=FLAGS.num_layers,
                dropout=FLAGS.dropout).to(device)
            model = FTModel(enc, num_classes, FLAGS.dropout).to(device)
            model = train_ft(model, data, y, tm, vm, tsm, device)
            model.eval()
            with torch.no_grad():
                logits = model(data)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds = logits.argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            overall_auc = roc_auc_score(y_np[tsm_np], probs[tsm_np]) * 100.0
            overall_f1 = f1_score(y_np[tsm_np], preds[tsm_np],
                                  average='macro', zero_division=0) * 100.0
            fair = compute_group_fairness(
                y_np, preds, probs, sens_np, test_mask=tsm_np)
            print(
                f"[FAIR_RAW] method={tag} dataset={FLAGS.dataset} "
                f"split={split_idx} seed={run_seed} "
                f"auc={overall_auc:.4f} macro_f1={overall_f1:.4f} "
                f"delta_sp={fair['delta_sp']:.4f} delta_eo={fair['delta_eo']:.4f} "
                f"delta_utility={fair['delta_utility']:.4f} "
                f"auc_s0={fair['auc_s0']} auc_s1={fair['auc_s1']} "
                f"n_s0={fair['n_s0']} n_s1={fair['n_s1']}")
            results.append({
                'split': split_idx, 'seed': run_seed,
                'auc': overall_auc, 'macro_f1': overall_f1,
                **fair,
            })

    def _agg(k):
        vals = [r[k] for r in results if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vals: return float('nan'), float('nan')
        return float(np.mean(vals)), float(np.std(vals))

    m_auc, s_auc = _agg('auc')
    m_f1, s_f1 = _agg('macro_f1')
    m_sp, s_sp = _agg('delta_sp')
    m_eo, s_eo = _agg('delta_eo')
    m_u, s_u = _agg('delta_utility')
    print(
        f"[FAIR_AGG] method={tag} dataset={FLAGS.dataset} "
        f"n_runs={len(results)} "
        f'auc="{m_auc:.2f} ± {s_auc:.2f}" '
        f'macro_f1="{m_f1:.2f} ± {s_f1:.2f}" '
        f'delta_sp="{m_sp:.4f} ± {s_sp:.4f}" '
        f'delta_eo="{m_eo:.4f} ± {s_eo:.4f}" '
        f'delta_utility="{m_u:.4f} ± {s_u:.4f}"')


if __name__ == "__main__":
    app.run(main)
