"""Step imbalance split generator.

Follows the convention used in TAM (ICML 2022) / ReNode (NeurIPS 2021):

  1. Identify |Y| classes. Sort by original training count ascending.
  2. Select the lower half (|Y|/2) as MINOR classes.
  3. Let n_major = max training-node count in the original train set.
  4. For each minor class, randomly drop labeled train nodes until exactly
     n_minor = floor(n_major / rho) remain.
  5. Graph, val, test are unchanged. Only train_mask is modified.

Deterministic given (dataset_hash, rho, seed).
"""

import torch
import numpy as np


def make_step_imbalance(train_mask, y, rho, seed=0, minor_frac=0.5):
    """Return a new train_mask with step imbalance applied.

    Args:
        train_mask: bool tensor [N], original train mask
        y: long tensor [N], node labels (ignore where train_mask is False)
        rho: int, imbalance ratio (n_major / n_minor). rho=1 means no change.
        seed: int, RNG seed for which minor-class nodes to keep
        minor_frac: float in (0,1), fraction of classes to treat as minor.
            Default 0.5 (half the classes) per TAM.

    Returns:
        new_train_mask: bool tensor [N], with minor classes downsampled.
        meta: dict with diagnostics (class counts before/after, minor class ids).
    """
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()

    y_cpu = y.detach().cpu().long()
    train_cpu = train_mask.detach().cpu()

    # Original per-class training counts
    train_idx = torch.nonzero(train_cpu, as_tuple=False).flatten()
    train_labels = y_cpu[train_idx]
    num_classes = int(y_cpu.max().item()) + 1

    class_counts = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        class_counts[c] = int((train_labels == c).sum().item())

    # Rank classes ascending by training count; select lower half as minor
    n_minor_classes = max(1, int(round(num_classes * minor_frac)))
    _, sorted_class_ids = torch.sort(class_counts, descending=False)
    minor_classes = set(sorted_class_ids[:n_minor_classes].tolist())
    major_classes = set(sorted_class_ids[n_minor_classes:].tolist())

    n_major_max = int(class_counts[list(major_classes)].max().item()) if major_classes else 0
    n_minor_target = max(1, int(n_major_max // rho))

    # Build new train mask
    new_train = torch.zeros_like(train_cpu)
    rng = np.random.RandomState(seed)

    for c in range(num_classes):
        idx_c = train_idx[train_labels == c].tolist()
        if c in minor_classes:
            # Keep only n_minor_target samples
            if len(idx_c) <= n_minor_target:
                keep = idx_c  # nothing to drop
            else:
                keep = rng.choice(idx_c, size=n_minor_target, replace=False).tolist()
            for i in keep:
                new_train[i] = True
        else:
            # Major class: keep original labels untouched
            for i in idx_c:
                new_train[i] = True

    # Diagnostics
    new_counts = torch.zeros(num_classes, dtype=torch.long)
    for c in range(num_classes):
        new_counts[c] = int((y_cpu[new_train] == c).sum().item())

    meta = {
        "rho": int(rho),
        "seed": int(seed),
        "num_classes": num_classes,
        "minor_classes": sorted(minor_classes),
        "major_classes": sorted(major_classes),
        "n_major_max": n_major_max,
        "n_minor_target": n_minor_target,
        "counts_before": class_counts.tolist(),
        "counts_after": new_counts.tolist(),
        "train_size_before": int(train_cpu.sum().item()),
        "train_size_after": int(new_train.sum().item()),
    }

    new_train = new_train.to(train_mask.device)
    return new_train, meta


def compute_imbalance_metrics(y_true, y_pred, num_classes=None):
    """Compute balanced accuracy + macro-F1 + per-class recall.

    Args:
        y_true: 1-D int array or tensor [N_test]
        y_pred: 1-D int array or tensor [N_test]
        num_classes: int, optional. If not given, inferred from y_true max+1.

    Returns:
        dict with keys: bacc, macro_f1, acc, per_class_recall (list), per_class_f1 (list)
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

    if hasattr(y_true, "cpu"):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    bacc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    acc = (y_true == y_pred).mean() * 100.0

    labels = list(range(num_classes))
    per_class_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ) * 100.0
    per_class_f1 = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    ) * 100.0

    return {
        "bacc": float(bacc),
        "macro_f1": float(macro_f1),
        "acc": float(acc),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }
