"""DeepWalk + degree-shift OOD eval (NC datasets).

Protocol mirrors gnn_baseline.run_ood_degree.py:
  - GOOD 60/20/20 descending degree split (build_degree_split).
  - Transductive: learn embedding on the FULL clean graph (including OOD nodes).
  - Fit sklearn LogisticRegression on train_mask only.
  - Predict id_test + ood_test + id_val + ood_val.
  - Emit [OOD_RAW] (selector=id_val) and [OOD_ORACLE] (selector=ood_val).

For shallow methods there is no per-epoch checkpoint to differentiate
id_val-best vs ood_val-best. We emit both lines with identical numbers
for collector compatibility.
"""

import os.path as osp
import sys

import numpy as np
import torch
from absl import app, flags
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from torch_geometric.nn import Node2Vec

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag


# --- Inlined from sgb/models/supervised/gnn_baseline/run_ood_degree.py to avoid absl
#     duplicate-flag collision when importing that module ---
def _compute_node_degree(edge_index, num_nodes):
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg


def build_degree_split(dataset_name, edge_index, labels, split_seed):
    """GOOD 60/20/20 descending degree split (mirrors run_ood_degree.py)."""
    labels_cpu = labels.detach().cpu().long()
    num_nodes = int(labels_cpu.numel())
    if labels_cpu.dtype.is_floating_point:
        labeled_bool = ~torch.isnan(labels_cpu)
    else:
        labeled_bool = labels_cpu >= 0
    labeled_idx_all = torch.arange(num_nodes)[labeled_bool]
    labeled_y_all = labels_cpu[labeled_idx_all]

    deg = _compute_node_degree(edge_index, num_nodes)
    labeled_deg = deg[labeled_idx_all].long()
    num_classes_total = int(torch.unique(labeled_y_all).numel())

    sort_key = labeled_deg * (num_nodes + 1) + labeled_idx_all.long()
    order = torch.argsort(sort_key, descending=True)
    sorted_idx = labeled_idx_all[order]
    sorted_y = labeled_y_all[order]
    sorted_deg = labeled_deg[order]

    n_labeled = int(sorted_idx.numel())
    train_end = int(round(n_labeled * 0.60))
    ood_val_end = int(round(n_labeled * 0.80))

    train_pool_idx = sorted_idx[:train_end]
    train_pool_y = sorted_y[:train_end]
    ood_val_idx = sorted_idx[train_end:ood_val_end]
    ood_test_idx = sorted_idx[ood_val_end:]

    if (train_pool_idx.numel() == 0 or ood_val_idx.numel() == 0
            or ood_test_idx.numel() == 0):
        return {
            "train": torch.empty(0, dtype=torch.long),
            "id_val": torch.empty(0, dtype=torch.long),
            "id_test": torch.empty(0, dtype=torch.long),
            "ood_val": torch.empty(0, dtype=torch.long),
            "ood_test": torch.empty(0, dtype=torch.long),
            "meta": {
                "dataset": dataset_name, "split_seed": split_seed,
                "shift": "degree", "strategy": "good_60_20_20_descending",
                "degree_shift": "not_applicable", "reason": "empty_bucket",
                "num_classes": int(num_classes_total),
                "num_nodes_total": int(num_nodes),
                "n_labeled": int(n_labeled),
            },
        }

    num_id = int(round(n_labeled * 0.10))
    if 2 * num_id >= train_pool_idx.numel():
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(
            rng.permutation(int(train_pool_idx.numel())), dtype=torch.long,
        )
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id: -num_id]
        id_test_idx = shuffled[-num_id:]

    train_pool_counts = torch.bincount(train_pool_y, minlength=num_classes_total)
    present = int((train_pool_counts > 0).sum().item())
    smallest = int(train_pool_counts[train_pool_counts > 0].min().item()) \
        if present > 0 else 0
    missing_classes = int(num_classes_total - present)

    def _range_tuple(pool_deg):
        if pool_deg.numel() == 0:
            return (None, None)
        return (int(pool_deg.min().item()), int(pool_deg.max().item()))

    return {
        "train": actual_train_idx,
        "id_val": id_val_idx,
        "id_test": id_test_idx,
        "ood_val": ood_val_idx,
        "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "shift": "degree",
            "strategy": "good_60_20_20_descending",
            "degree_shift": "ok",
            "num_classes": int(num_classes_total),
            "num_nodes_total": int(num_nodes),
            "n_labeled": int(n_labeled),
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()),
            "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()),
            "ood_test_size": int(ood_test_idx.numel()),
            "train_pool_degree_range": _range_tuple(sorted_deg[:train_end]),
            "ood_val_degree_range": _range_tuple(sorted_deg[train_end:ood_val_end]),
            "ood_test_degree_range": _range_tuple(sorted_deg[ood_val_end:]),
            "smallest_train_pool_class": int(smallest),
            "missing_classes_in_train_pool": int(missing_classes),
        },
    }


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG NC dataset name.')
flags.DEFINE_integer('emb_dim', 128, 'Embedding dim.')
flags.DEFINE_integer('walk_length', 20, 'Random walk length.')
flags.DEFINE_integer('context_size', 10, 'Skip-gram context size.')
flags.DEFINE_integer('walks_per_node', 10, 'Walks per node.')
flags.DEFINE_integer('num_neg', 1, 'Negative samples per positive.')
flags.DEFINE_float('p', 1.0, 'Return parameter.')
flags.DEFINE_float('q', 0.5, 'In-out parameter (q<1 favors BFS / homophilous community).')
flags.DEFINE_integer('walk_epochs', 100, 'Skip-gram training epochs.')
flags.DEFINE_integer('walk_batch_size', 128, 'Walk batch size.')
flags.DEFINE_float('walk_lr', 1e-2, 'SparseAdam lr.')
flags.DEFINE_integer('num_workers', 0, 'DataLoader workers.')
flags.DEFINE_float('lr_C', 1.0, 'sklearn LR C.')
flags.DEFINE_integer('lr_max_iter', 1000, 'sklearn LR max_iter.')
flags.DEFINE_integer('num_splits', 5, 'Number of split seeds (clamp <=5).')

METHOD_TAG = "Node2Vec"

SPLIT_SEEDS = [0, 1, 2, 3, 4]


def learn_embedding(edge_index, num_nodes, device):
    model = Node2Vec(
        edge_index,
        embedding_dim=FLAGS.emb_dim,
        walk_length=FLAGS.walk_length,
        context_size=FLAGS.context_size,
        walks_per_node=FLAGS.walks_per_node,
        num_negative_samples=FLAGS.num_neg,
        p=FLAGS.p,
        q=FLAGS.q,
        num_nodes=num_nodes,
        sparse=True,
    ).to(device)
    loader = model.loader(batch_size=FLAGS.walk_batch_size, shuffle=True,
                          num_workers=FLAGS.num_workers)
    optim = torch.optim.SparseAdam(list(model.parameters()), lr=FLAGS.walk_lr)
    model.train()
    for epoch in range(1, FLAGS.walk_epochs + 1):
        total = 0.0
        n_batches = 0
        for pos_rw, neg_rw in loader:
            optim.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optim.step()
            total += float(loss.item())
            n_batches += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == FLAGS.walk_epochs:
            avg = total / max(1, n_batches)
            print(f"    [walk] epoch {epoch:>3}/{FLAGS.walk_epochs}  loss={avg:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        emb = model().detach().cpu().numpy()
    return emb


def _eval_subset(emb, y_np, train_mask_np, mask_np, clf):
    """Return (acc, f1) on given mask using already-fit classifier."""
    if mask_np.sum() == 0:
        return float('nan'), float('nan')
    pred = clf.predict(emb[mask_np])
    y_true = y_np[mask_np]
    acc = float((pred == y_true).mean()) * 100.0
    macro_f1 = f1_score(y_true, pred, average='macro') * 100.0
    return acc, macro_f1


def _gap(id_v, ood_v):
    if id_v is None or ood_v is None or np.isnan(id_v) or np.isnan(ood_v):
        return float('nan'), float('nan'), float('nan')
    gap_abs = id_v - ood_v
    gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
    rr = ood_v / id_v if id_v > 0 else 0.0
    return gap_abs, gap_rel, rr


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[{METHOD_TAG} OOD-DEGREE] Using {device}, dataset={FLAGS.dataset}", flush=True)

    data, _ = load_tag(FLAGS.dataset)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    y_np = data.y.cpu().numpy()
    N = int(data.num_nodes)
    edge_index = data.edge_index.clone().to(device)
    num_classes = int(data.y.max().item()) + 1

    print(f"[{METHOD_TAG} OOD-DEGREE] {FLAGS.dataset}, N={N}, "
          f"E={edge_index.size(1)}, C={num_classes}", flush=True)

    n_split = min(FLAGS.num_splits, 5)
    split_seeds = SPLIT_SEEDS[:n_split]

    for split_seed in split_seeds:
        # Match edge-deletion / fairness convention: seed = split_seed, set both RNGs.
        torch.manual_seed(split_seed)
        np.random.seed(split_seed)

        five_way = build_degree_split(
            dataset_name=FLAGS.dataset,
            edge_index=data.edge_index,  # CPU tensor for numpy ops in builder
            labels=data.y,
            split_seed=split_seed,
        )
        meta = five_way["meta"]
        if meta.get("degree_shift") == "not_applicable":
            print(f"[OOD_SKIP] method={METHOD_TAG} dataset={FLAGS.dataset} "
                  f"split_seed={split_seed} reason={meta.get('reason', 'unknown')}",
                  flush=True)
            continue

        print(
            f"[OOD_SPLIT] dataset={FLAGS.dataset} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_20_20_descending')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_deg_range={meta['train_pool_degree_range']} "
            f"ood_test_deg_range={meta['ood_test_degree_range']}",
            flush=True,
        )

        train_idx = five_way["train"]
        id_val_idx = five_way["id_val"]
        id_test_idx = five_way["id_test"]
        ood_val_idx = five_way["ood_val"]
        ood_test_idx = five_way["ood_test"]

        train_mask = np.zeros(N, dtype=bool); train_mask[train_idx.cpu().numpy()] = True
        id_val_mask = np.zeros(N, dtype=bool); id_val_mask[id_val_idx.cpu().numpy()] = True
        id_test_mask = np.zeros(N, dtype=bool); id_test_mask[id_test_idx.cpu().numpy()] = True
        ood_val_mask = np.zeros(N, dtype=bool); ood_val_mask[ood_val_idx.cpu().numpy()] = True
        ood_test_mask = np.zeros(N, dtype=bool); ood_test_mask[ood_test_idx.cpu().numpy()] = True

        emb = learn_embedding(edge_index, N, device)

        clf = LogisticRegression(
            C=FLAGS.lr_C,
            max_iter=FLAGS.lr_max_iter,
            n_jobs=-1,
            solver='lbfgs',
        )
        clf.fit(emb[train_mask], y_np[train_mask])

        id_val_acc, _ = _eval_subset(emb, y_np, train_mask, id_val_mask, clf)
        id_test_acc, id_test_f1 = _eval_subset(emb, y_np, train_mask, id_test_mask, clf)
        ood_val_acc, _ = _eval_subset(emb, y_np, train_mask, ood_val_mask, clf)
        ood_test_acc, ood_test_f1 = _eval_subset(emb, y_np, train_mask, ood_test_mask, clf)

        gap_abs, gap_rel, rr = _gap(id_test_acc, ood_test_acc)

        # selector=id_val (main) and selector=ood_val (oracle).
        # Shallow has no per-epoch checkpoint, so both report identical numbers.
        for selector_tag, selector_name in [("OOD_RAW", "id_val"),
                                            ("OOD_ORACLE", "ood_val")]:
            print(
                f"[{selector_tag}] method={METHOD_TAG} dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={split_seed} "
                f"shift=degree selector={selector_name} "
                f"id={id_test_acc:.4f} ood={ood_test_acc:.4f} "
                f"gap_abs={gap_abs:.4f} gap_rel={gap_rel:.4f} rr={rr:.4f} "
                f"id_val={id_val_acc:.4f} ood_val={ood_val_acc:.4f} "
                f"id_test_f1={id_test_f1:.4f} ood_test_f1={ood_test_f1:.4f}",
                flush=True,
            )


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
