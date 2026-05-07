"""DeepWalk + temporal-shift OOD eval (arxiv only).

Protocol mirrors gnn_baseline.run_ood_time.py:
  - Covariate shift along publication year (oldest 60% train, newest 35% ood_test).
  - Transductive: learn embedding on full clean graph (all years).
  - Fit sklearn LogisticRegression on train mask only.
  - Predict id_test + ood_test + id_val + ood_val.
  - Emit [OOD_RAW] (selector=id_val) and [OOD_ORACLE] (selector=ood_val).

For shallow methods there is no per-epoch checkpoint, so OOD_RAW and
OOD_ORACLE report identical numbers (kept for collector compatibility).

n=5 runs: each iteration uses seed=i for split permutation + walk randomness.
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
from sgb.data.ood_splits import build_time_shift_split

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'arxiv', 'TAG dataset (must have node_year).')
flags.DEFINE_integer('emb_dim', 128, 'Embedding dim.')
flags.DEFINE_integer('walk_length', 20, 'Random walk length.')
flags.DEFINE_integer('context_size', 10, 'Skip-gram context size.')
flags.DEFINE_integer('walks_per_node', 10, 'Walks per node.')
flags.DEFINE_integer('num_neg', 1, 'Negative samples per positive.')
flags.DEFINE_float('p', 1.0, 'Return parameter.')
flags.DEFINE_float('q', 1.0, 'In-out parameter.')
flags.DEFINE_integer('walk_epochs', 100, 'Skip-gram training epochs.')
flags.DEFINE_integer('walk_batch_size', 128, 'Walk batch size.')
flags.DEFINE_float('walk_lr', 1e-2, 'SparseAdam lr.')
flags.DEFINE_integer('num_workers', 0, 'DataLoader workers.')
flags.DEFINE_float('lr_C', 1.0, 'sklearn LR C.')
flags.DEFINE_integer('lr_max_iter', 1000, 'sklearn LR max_iter.')
flags.DEFINE_integer('num_splits', 5, 'Number of seeds (n=5 runs).')
flags.DEFINE_integer('train_max_year', -1,
                     'If >0, train_pool restricted to year <= this. Else position-based 60/35.')
flags.DEFINE_integer('ood_min_year', -1,
                     'If >0, ood_test restricted to year >= this. Pairs with train_max_year.')

METHOD_TAG = "DeepWalk"


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


def _eval_subset(emb, y_np, mask_np, clf):
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
    print(f"[{METHOD_TAG} OOD-TIME] Using {device}, dataset={FLAGS.dataset}", flush=True)

    data, _ = load_tag(FLAGS.dataset)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    if not hasattr(data, 'node_year') or data.node_year is None:
        raise RuntimeError(
            f"Dataset {FLAGS.dataset} has no node_year; time shift not applicable.")

    y_np = data.y.cpu().numpy()
    N = int(data.num_nodes)
    edge_index = data.edge_index.clone().to(device)
    year = data.node_year
    num_classes = int(data.y.max().item()) + 1

    print(f"[{METHOD_TAG} OOD-TIME] {FLAGS.dataset}, N={N}, "
          f"E={edge_index.size(1)}, C={num_classes}", flush=True)

    train_max_year = FLAGS.train_max_year if FLAGS.train_max_year > 0 else None
    ood_min_year = FLAGS.ood_min_year if FLAGS.ood_min_year > 0 else None

    n_split = min(FLAGS.num_splits, 5)

    for split_seed in range(n_split):
        torch.manual_seed(split_seed)
        np.random.seed(split_seed)

        five_way = build_time_shift_split(
            dataset_name=FLAGS.dataset,
            year_tensor=year,
            labels=data.y,
            split_seed=split_seed,
            train_max_year=train_max_year,
            ood_min_year=ood_min_year,
        )
        meta = five_way["meta"]
        if meta.get("time_shift") == "not_applicable":
            print(f"[OOD_SKIP] method={METHOD_TAG} dataset={FLAGS.dataset} "
                  f"split_seed={split_seed} reason={meta.get('reason', 'unknown')}",
                  flush=True)
            continue

        print(
            f"[OOD_SPLIT] dataset={FLAGS.dataset} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'time')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_year_range={meta.get('train_pool_year_range')} "
            f"ood_test_year_range={meta.get('ood_test_year_range')}",
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
            C=FLAGS.lr_C, max_iter=FLAGS.lr_max_iter,
            n_jobs=-1, solver='lbfgs',
        )
        clf.fit(emb[train_mask], y_np[train_mask])

        id_val_acc, _ = _eval_subset(emb, y_np, id_val_mask, clf)
        id_test_acc, id_test_f1 = _eval_subset(emb, y_np, id_test_mask, clf)
        ood_val_acc, _ = _eval_subset(emb, y_np, ood_val_mask, clf)
        ood_test_acc, ood_test_f1 = _eval_subset(emb, y_np, ood_test_mask, clf)

        gap_abs, gap_rel, rr = _gap(id_test_acc, ood_test_acc)

        for selector_tag, selector_name in [("OOD_RAW", "id_val"),
                                            ("OOD_ORACLE", "ood_val")]:
            print(
                f"[{selector_tag}] method={METHOD_TAG} dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={split_seed} "
                f"shift=time selector={selector_name} "
                f"id={id_test_acc:.4f} ood={ood_test_acc:.4f} "
                f"gap_abs={gap_abs:.4f} gap_rel={gap_rel:.4f} rr={rr:.4f} "
                f"id_val={id_val_acc:.4f} ood_val={ood_val_acc:.4f} "
                f"id_test_f1={id_test_f1:.4f} ood_test_f1={ood_test_f1:.4f}",
                flush=True,
            )


if __name__ == '__main__':
    app.run(main)
