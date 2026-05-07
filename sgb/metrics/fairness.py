"""Fairness split + metrics for GFM-Safety extension.

Two lenses (complementary, both belong under "Fairness" umbrella):

1. **Demographic fairness** (tolokers): sensitive attribute = education binary
   (high vs not-high). Metrics follow PyGDebias / FairGNN standard:
     - Delta_SP  = |P(Y_hat=1 | S=0) - P(Y_hat=1 | S=1)|
     - Delta_EO  = |P(Y_hat=1 | Y=1, S=0) - P(Y_hat=1 | Y=1, S=1)|
     - Delta_Utility = |AUC(S=0) - AUC(S=1)|
   Report mean ± std over runs.

2. **Structural fairness** (any node dataset): sensitive group = node degree
   bucket (top-20% / bottom-20%). Measures performance parity across the
   structural axis. Standard (Liu 2022, Kang 2020).
     - Acc_head - Acc_tail
     - F1_head  - F1_tail

Both are POST-HOC on existing predictions — no re-training required beyond
saving `y_true`, `y_pred`, `sensitive` arrays.
"""

import numpy as np
import torch


def load_tolokers_education_binary():
    """Load tolokers' education attribute from raw nodes.tsv.

    Returns:
        sensitive: long tensor [N], 1 = high-education, 0 = not
        meta: dict with class counts
    """
    import os
    _here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.environ.get(
        'GFM_PROJECT_ROOT',
        os.path.abspath(os.path.join(_here, '..', '..')),
    )
    # tolokers raw nodes live under datasets/TAG_raw/tolokers/nodes.tsv
    candidates = [
        os.path.join(project_root, 'sgb/data/dataset/data/single_graph/tolokers/nodes.tsv'),
        os.path.join(project_root, 'datasets/TAG_raw/tolokers/nodes.tsv'),
        os.path.join(project_root, 'datasets/TAG/tolokers/raw/nodes.tsv'),
        os.path.join(project_root, 'datasets/TAG/tolokers/nodes.tsv'),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            f"tolokers nodes.tsv not found; tried: {candidates}")

    import csv
    sens = []
    with open(path, 'r') as f:
        # Columns: id, approved_rate, skipped_rate, expired_rate, rejected_rate,
        # education, english_profile, english_tested, banned
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            edu = (row.get('education') or '').strip().lower()
            sens.append(1 if edu == 'high' else 0)

    s = torch.as_tensor(sens, dtype=torch.long)
    counts = {
        'n_total': int(s.numel()),
        'n_high': int((s == 1).sum().item()),
        'n_not_high': int((s == 0).sum().item()),
    }
    return s, counts


def compute_group_fairness(y_true, y_pred, y_score, sensitive, test_mask=None):
    """Delta_SP / Delta_EO / Delta_Utility per PyGDebias standard.

    Args:
        y_true: np.int array [N]
        y_pred: np.int array [N] (hard predictions)
        y_score: np.float array [N] (positive-class probability)
        sensitive: np.int array [N] (0 / 1)
        test_mask: optional bool mask of which nodes to include

    Returns:
        dict
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    sensitive = np.asarray(sensitive).astype(int)

    if test_mask is not None:
        tm = np.asarray(test_mask).astype(bool)
        y_true = y_true[tm]
        y_pred = y_pred[tm]
        y_score = y_score[tm]
        sensitive = sensitive[tm]

    s0 = sensitive == 0
    s1 = sensitive == 1

    # Delta SP
    p1_s0 = y_pred[s0].mean() if s0.sum() > 0 else 0.0
    p1_s1 = y_pred[s1].mean() if s1.sum() > 0 else 0.0
    delta_sp = abs(p1_s0 - p1_s1)

    # Delta EO (only among y_true=1)
    y1 = y_true == 1
    y1_s0 = y1 & s0
    y1_s1 = y1 & s1
    p_y1_s0 = y_pred[y1_s0].mean() if y1_s0.sum() > 0 else 0.0
    p_y1_s1 = y_pred[y1_s1].mean() if y1_s1.sum() > 0 else 0.0
    delta_eo = abs(p_y1_s0 - p_y1_s1)

    # Delta Utility (AUC per group)
    def _safe_auc(y_t, y_s):
        if len(np.unique(y_t)) < 2:
            return float('nan')
        return roc_auc_score(y_t, y_s)

    auc_s0 = _safe_auc(y_true[s0], y_score[s0])
    auc_s1 = _safe_auc(y_true[s1], y_score[s1])
    if not np.isnan(auc_s0) and not np.isnan(auc_s1):
        delta_util = abs(auc_s0 - auc_s1)
    else:
        delta_util = float('nan')

    return {
        'delta_sp': float(delta_sp),
        'delta_eo': float(delta_eo),
        'delta_utility': float(delta_util),
        'auc_s0': float(auc_s0) if not np.isnan(auc_s0) else None,
        'auc_s1': float(auc_s1) if not np.isnan(auc_s1) else None,
        'n_s0': int(s0.sum()), 'n_s1': int(s1.sum()),
    }


def compute_structural_fairness(y_true, y_pred, degree, test_mask=None, q=0.2):
    """Per-degree-group accuracy and F1 gap.

    head = top-q by degree, tail = bottom-q. Smaller gap = fairer.
    """
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    degree = np.asarray(degree).astype(float)

    if test_mask is not None:
        tm = np.asarray(test_mask).astype(bool)
        y_true = y_true[tm]
        y_pred = y_pred[tm]
        degree = degree[tm]

    if len(degree) == 0:
        return {'acc_head': float('nan'), 'acc_tail': float('nan'),
                'acc_gap': float('nan'), 'f1_head': float('nan'),
                'f1_tail': float('nan'), 'f1_gap': float('nan')}

    d_sorted = np.sort(degree)
    n = len(d_sorted)
    tail_thr = d_sorted[int(q * n)]
    head_thr = d_sorted[int((1 - q) * n)]

    tail = degree <= tail_thr
    head = degree >= head_thr

    def _acc(m):
        if m.sum() == 0: return float('nan')
        return (y_true[m] == y_pred[m]).mean() * 100.0

    def _f1(m):
        if m.sum() == 0: return float('nan')
        return f1_score(y_true[m], y_pred[m], average='macro', zero_division=0) * 100.0

    acc_head, acc_tail = _acc(head), _acc(tail)
    f1_head, f1_tail = _f1(head), _f1(tail)

    # ---- Per-degree-bin diagnostic (Tang CIKM'20 Fig 8 grammar) ----
    # rank-based 5-quantile split (robust to degree ties)
    n_bins = 5
    order = np.argsort(degree, kind='stable')
    bin_idx_sets = np.array_split(order, n_bins)
    fixed_labels = np.unique(y_true)  # comparable F1 across bins

    bin_stats = []
    for b, idx in enumerate(bin_idx_sets):
        if len(idx) == 0:
            bin_stats.append({'bin': b, 'lo': float('nan'), 'hi': float('nan'),
                              'n': 0, 'acc': float('nan'), 'f1': float('nan')})
            continue
        d_b = degree[idx]
        y_b, p_b = y_true[idx], y_pred[idx]
        bin_acc = float((y_b == p_b).mean() * 100.0)
        bin_f1 = float(f1_score(y_b, p_b, labels=fixed_labels,
                                average='macro', zero_division=0) * 100.0)
        bin_stats.append({'bin': b, 'lo': float(d_b.min()), 'hi': float(d_b.max()),
                          'n': int(len(idx)), 'acc': bin_acc, 'f1': bin_f1})

    # Print parseable line; aggregator binds it to the immediately
    # preceding [STRUCT_RAW] line which carries (method, dataset, split, seed).
    print('[STRUCT_BIN] ' + ' '.join(
        f"bin{s['bin']}=acc:{s['acc']:.4f},f1:{s['f1']:.4f},"
        f"n:{s['n']},lo:{s['lo']:.2f},hi:{s['hi']:.2f}"
        for s in bin_stats), flush=True)

    return {
        'acc_head': float(acc_head), 'acc_tail': float(acc_tail),
        'acc_gap': float(acc_head - acc_tail) if not (np.isnan(acc_head) or np.isnan(acc_tail)) else float('nan'),
        'f1_head': float(f1_head), 'f1_tail': float(f1_tail),
        'f1_gap': float(f1_head - f1_tail) if not (np.isnan(f1_head) or np.isnan(f1_tail)) else float('nan'),
        'n_head': int(head.sum()), 'n_tail': int(tail.sum()),
        'per_bin': bin_stats,
    }
