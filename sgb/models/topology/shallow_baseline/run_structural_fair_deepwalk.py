"""DeepWalk + structural (degree-based) fairness eval (NC datasets).

Protocol:
  - Train DeepWalk on the clean graph (no perturbation), per split.
  - Fit sklearn LogisticRegression on emb[train_mask].
  - Predict on test_mask.
  - Compute structural fairness (head/tail by degree quantile q=0.2).
  - n=5 runs (seed = split_idx), mirror gnn_baseline convention.

Emits [STRUCT_RAW] per split + [STRUCT_AGG] aggregated. Collector reads AGG.
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
from sgb.metrics.fairness import compute_structural_fairness

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG NC dataset name.')
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
flags.DEFINE_float('q_quantile', 0.2, 'Degree quantile for head/tail split.')
flags.DEFINE_integer('num_splits', 5, 'Number of splits (clamp <=5).')

METHOD_TAG = "DeepWalk"


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def build_splits(data):
    """5 splits (mirror gnn_baseline.finetune_edge_deletion / structural_fair)."""
    splits = []
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        for i in range(5):
            j = i % avail
            splits.append({
                'train': data.train_masks[j].bool(),
                'val': data.val_masks[j].bool(),
                'test': data.test_masks[j].bool(),
            })
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        N = data.num_nodes
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
        for _ in range(5):
            splits.append({'train': tm, 'val': vm, 'test': tsm})
    else:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            avail = tm.size(1)
            for i in range(5):
                j = i % avail
                splits.append({
                    'train': tm[:, j].bool(),
                    'val': vm[:, j].bool(),
                    'test': (tsm[:, j] if tsm.dim() == 2 else tsm).bool(),
                })
        else:
            for _ in range(5):
                splits.append({'train': tm.bool(), 'val': vm.bool(), 'test': tsm.bool()})
    return splits


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


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[{METHOD_TAG} STRUCT] Using {device}, dataset={FLAGS.dataset}", flush=True)

    data, _ = load_tag(FLAGS.dataset)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    y_np = data.y.cpu().numpy()
    N = int(data.num_nodes)
    edge_index = data.edge_index.clone().to(device)

    # Mirror gnn_baseline.finetune_structural_fair: bincount on src endpoint.
    degree = torch.bincount(data.edge_index[0], minlength=N).cpu().numpy()
    print(f"[{METHOD_TAG} STRUCT] N={N}, E={edge_index.size(1)}, "
          f"deg mean={degree.mean():.1f} max={degree.max()} min={degree.min()} "
          f"q={FLAGS.q_quantile}", flush=True)

    splits = build_splits(data)
    n_split = min(FLAGS.num_splits, 5)

    results = []
    for split_idx in range(n_split):
        # Aligned with edge-deletion convention: seed = split_idx, set both RNGs.
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        split = splits[split_idx]
        train_mask_np = split['train'].cpu().numpy().astype(bool)
        test_mask_np = split['test'].cpu().numpy().astype(bool)

        emb = learn_embedding(edge_index, N, device)

        clf = LogisticRegression(
            C=FLAGS.lr_C,
            max_iter=FLAGS.lr_max_iter,
            n_jobs=-1,
            solver='lbfgs',
        )
        clf.fit(emb[train_mask_np], y_np[train_mask_np])
        y_pred = clf.predict(emb[test_mask_np])
        y_true_np = y_np[test_mask_np]

        # Build full-length pred array for compute_structural_fairness API.
        full_pred = np.zeros(N, dtype=y_np.dtype)
        full_pred[test_mask_np] = y_pred
        struct = compute_structural_fairness(
            y_np, full_pred, degree,
            test_mask=test_mask_np, q=FLAGS.q_quantile)

        acc = float((y_pred == y_true_np).mean()) * 100.0
        print(f"[STRUCT_RAW] method={METHOD_TAG} dataset={FLAGS.dataset} "
              f"split={split_idx} seed={split_idx} acc={acc:.4f} "
              f"acc_head={struct['acc_head']:.4f} acc_tail={struct['acc_tail']:.4f} "
              f"acc_gap={struct['acc_gap']:.4f} "
              f"f1_head={struct['f1_head']:.4f} f1_tail={struct['f1_tail']:.4f} "
              f"f1_gap={struct['f1_gap']:.4f} "
              f"n_head={struct['n_head']} n_tail={struct['n_tail']} "
              f"q={FLAGS.q_quantile}", flush=True)
        results.append({'acc': acc, **struct})

    def _agg(k):
        vs = [r[k] for r in results
              if r[k] is not None
              and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs:
            return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))

    a, sa = _agg('acc')
    ah, sah = _agg('acc_head')
    at, sat = _agg('acc_tail')
    ag, sag = _agg('acc_gap')
    fh, sfh = _agg('f1_head')
    ft, sft = _agg('f1_tail')
    fg, sfg = _agg('f1_gap')
    print(f'[STRUCT_AGG] method={METHOD_TAG} dataset={FLAGS.dataset} '
          f'n_runs={len(results)} '
          f'acc="{a:.2f} ± {sa:.2f}" '
          f'acc_head="{ah:.2f} ± {sah:.2f}" acc_tail="{at:.2f} ± {sat:.2f}" '
          f'acc_gap="{ag:.2f} ± {sag:.2f}" '
          f'f1_head="{fh:.2f} ± {sfh:.2f}" f1_tail="{ft:.2f} ± {sft:.2f}" '
          f'f1_gap="{fg:.2f} ± {sfg:.2f}" q={FLAGS.q_quantile}', flush=True)


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
