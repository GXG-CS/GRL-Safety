"""DeepWalk + step-imbalance eval (NC datasets).

Protocol mirrors gnn_baseline.run_imbalance.py (TAM 2022 / ReNode 2021):
  - For each rep: apply step imbalance to train mask (subsample minor classes).
  - Eval on (clean) val/test.
  - Report bAcc, macro-F1, per-class recall + f1.

Shallow optimization: embedding is unsupervised (label-free), so we LEARN
ONE EMBEDDING PER REP and reuse it across all rho values. This is N×3 cheaper
than the GNN baseline which retrains for each rho.

Outputs [IMB_RAW] / [IMB_PER_CLASS] / [IMB_AGG] schema-compatible with
existing collector.
"""

import os.path as osp
import sys

import numpy as np
import torch
from absl import app, flags
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import Node2Vec

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance, compute_imbalance_metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG NC dataset name.')
flags.DEFINE_string('rhos', '5,10,20', 'Comma-separated imbalance ratios.')
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
flags.DEFINE_integer('n_reps', 5, 'Number of repetitions (paper-wide n=5).')

METHOD_TAG = "DeepWalk"


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_base_masks(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        return (data.train_masks[0].bool(), data.val_masks[0].bool(),
                data.test_masks[0].bool())
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits; N = data.num_nodes
        return (_idx_to_mask(s['train'], N),
                _idx_to_mask(s.get('valid', s.get('val')), N),
                _idx_to_mask(s['test'], N))
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        return (tm[:, 0].bool(), vm[:, 0].bool(),
                (tsm[:, 0] if tsm.dim() == 2 else tsm).bool())
    return tm.bool(), vm.bool(), tsm.bool()


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
        total = 0.0; n_b = 0
        for pos_rw, neg_rw in loader:
            optim.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward(); optim.step()
            total += float(loss.item()); n_b += 1
        if epoch == 1 or epoch % 25 == 0 or epoch == FLAGS.walk_epochs:
            avg = total / max(1, n_b)
            print(f"    [walk] epoch {epoch:>3}/{FLAGS.walk_epochs}  loss={avg:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        emb = model().detach().cpu().numpy()
    return emb


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[{METHOD_TAG} IMB] Using {device}, dataset={FLAGS.dataset}", flush=True)

    rhos = [int(r.strip()) for r in FLAGS.rhos.split(',') if r.strip()]
    print(f"[{METHOD_TAG} IMB] rhos={rhos}, n_reps={FLAGS.n_reps}", flush=True)

    data, _ = load_tag(FLAGS.dataset)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    y = data.y
    y_np = y.cpu().numpy()
    N = int(data.num_nodes)
    edge_index = data.edge_index.clone().to(device)
    num_classes = int(y.max().item()) + 1

    train_base, val_mask, test_mask = _get_base_masks(data)
    train_base_np = train_base.cpu().numpy().astype(bool)
    test_mask_np = test_mask.cpu().numpy().astype(bool)

    print(f"[{METHOD_TAG} IMB] N={N}, E={edge_index.size(1)}, C={num_classes}", flush=True)
    print(f"  base train={int(train_base.sum())}, val={int(val_mask.sum())}, test={int(test_mask.sum())}", flush=True)

    # Per-(rho, rep) results, grouped by rho for AGG.
    grouped = {rho: [] for rho in rhos}

    for rep_idx in range(FLAGS.n_reps):
        seed = rep_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ONE embedding per rep, reused across rhos.
        emb = learn_embedding(edge_index, N, device)

        for rho in rhos:
            imb_train_mask, meta = make_step_imbalance(
                train_mask=train_base, y=y, rho=rho, seed=seed,
            )
            imb_train_np = imb_train_mask.cpu().numpy().astype(bool)

            if rep_idx == 0:
                print(f"  [rep 0 rho {rho}] n_major_max={meta['n_major_max']} "
                      f"n_minor_target={meta['n_minor_target']} "
                      f"minor_classes={meta['minor_classes']}", flush=True)

            clf = LogisticRegression(
                C=FLAGS.lr_C, max_iter=FLAGS.lr_max_iter,
                n_jobs=-1, solver='lbfgs',
            )
            clf.fit(emb[imb_train_np], y_np[imb_train_np])
            y_pred = clf.predict(emb[test_mask_np])
            y_true = y_np[test_mask_np]

            metrics = compute_imbalance_metrics(y_true, y_pred,
                                                num_classes=num_classes)
            grouped[rho].append({"rep": rep_idx, **metrics, "meta": meta})

            print(
                f"[IMB_RAW] method={METHOD_TAG} dataset={FLAGS.dataset} rho={rho} "
                f"rep={rep_idx} seed={seed} "
                f"bacc={metrics['bacc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
                f"acc={metrics['acc']:.4f} "
                f"n_minor_target={meta['n_minor_target']}",
                flush=True,
            )
            print(
                f"[IMB_PER_CLASS] method={METHOD_TAG} dataset={FLAGS.dataset} rho={rho} "
                f"rep={rep_idx} minor_classes={meta['minor_classes']} "
                f"per_class_recall={metrics['per_class_recall']} "
                f"per_class_f1={metrics['per_class_f1']}",
                flush=True,
            )

    for rho in rhos:
        results = grouped[rho]
        bacc = np.array([r['bacc'] for r in results])
        f1 = np.array([r['macro_f1'] for r in results])
        acc = np.array([r['acc'] for r in results])
        print(f"\n=== {METHOD_TAG} Imbalance Results ({FLAGS.dataset}, rho={rho}, n={len(results)}) ===", flush=True)
        print(f"  bAcc    = {bacc.mean():.2f} ± {bacc.std():.2f}", flush=True)
        print(f"  macroF1 = {f1.mean():.2f} ± {f1.std():.2f}", flush=True)
        print(f"  acc     = {acc.mean():.2f} ± {acc.std():.2f}", flush=True)
        print(
            f"[IMB_AGG] method={METHOD_TAG} dataset={FLAGS.dataset} rho={rho} "
            f"n_reps={len(results)} "
            f"bacc=\"{bacc.mean():.2f} ± {bacc.std():.2f}\" "
            f"macro_f1=\"{f1.mean():.2f} ± {f1.std():.2f}\" "
            f"acc=\"{acc.mean():.2f} ± {acc.std():.2f}\"",
            flush=True,
        )


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
