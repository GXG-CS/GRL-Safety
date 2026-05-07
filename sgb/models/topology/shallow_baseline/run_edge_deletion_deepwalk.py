"""Node2Vec edge-deletion eval (NC datasets).

Protocol:
  for each (dataset, split_idx) and each severity p in {0, 0.05, 0.10, 0.20, 0.30, 0.50}:
    1. corrupted_edges = drop_undirected_edges(clean_edges, p)
    2. learn Node2Vec embedding on corrupted_edges (100 walk-epochs)
    3. fit sklearn LogisticRegression on emb[train] -> y[train]
    4. report acc + macro_f1 on emb[test]

Topology-only baseline: no inductive forward, embeddings re-learned per topology.
"""

import collections
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

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG NC dataset name.')
flags.DEFINE_integer('emb_dim', 128, 'Node2Vec embedding dim.')
flags.DEFINE_integer('walk_length', 20, 'Random walk length.')
flags.DEFINE_integer('context_size', 10, 'Skip-gram context window.')
flags.DEFINE_integer('walks_per_node', 10, 'Walks per node.')
flags.DEFINE_integer('num_neg', 1, 'Negative samples per positive.')
flags.DEFINE_float('p', 1.0, 'Return parameter.')
flags.DEFINE_float('q', 1.0, 'In-out parameter.')
flags.DEFINE_integer('walk_epochs', 100, 'Skip-gram training epochs.')
flags.DEFINE_integer('walk_batch_size', 128, 'Walk batch size.')
flags.DEFINE_float('walk_lr', 1e-2, 'SparseAdam lr for embeddings.')
flags.DEFINE_integer('num_workers', 0, 'DataLoader workers (0 = main proc).')
flags.DEFINE_float('lr_C', 1.0, 'sklearn LogisticRegression C.')
flags.DEFINE_integer('lr_max_iter', 1000, 'sklearn LogisticRegression max_iter.')

METHOD_TAG = "DeepWalk"

SEVERITIES = [
    (1, 0.05),
    (2, 0.10),
    (3, 0.20),
    (4, 0.30),
    (5, 0.50),
]


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def apply_edge_drop(edge_index, num_nodes, p):
    """Drop edges at undirected-pair granularity (matches gnn_baseline)."""
    if p <= 0.0 or edge_index.size(1) == 0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    key = u.long() * num_nodes + v.long()
    _, inverse = torch.unique(key, return_inverse=True)
    num_undirected = int(inverse.max().item()) + 1
    keep = (torch.rand(num_undirected, device=edge_index.device) >= p)[inverse]
    keep = keep | (src == dst)
    return edge_index[:, keep]


def build_splits(data):
    """Mirrors gnn_baseline.finetune_edge_deletion.build_splits (5 splits)."""
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


def learn_node2vec(edge_index, num_nodes, device):
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

    loader = model.loader(
        batch_size=FLAGS.walk_batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
    )
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
            print(f"    [n2v] epoch {epoch:>3}/{FLAGS.walk_epochs}  loss={avg:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        emb = model().detach().cpu().numpy()
    return emb


def eval_lr(emb, y_np, train_mask_np, test_mask_np):
    clf = LogisticRegression(
        C=FLAGS.lr_C,
        max_iter=FLAGS.lr_max_iter,
        n_jobs=-1,
        solver='lbfgs',
    )
    clf.fit(emb[train_mask_np], y_np[train_mask_np])
    pred = clf.predict(emb[test_mask_np])
    y_true = y_np[test_mask_np]
    acc = float((pred == y_true).mean()) * 100.0
    macro_f1 = f1_score(y_true, pred, average='macro') * 100.0
    return acc, macro_f1


def aggregate_and_print(all_results, dataset):
    print(f"\n=== {METHOD_TAG} Edge Deletion Results ({dataset}) ===")
    grouped_acc = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped_acc[r['sev']].append(r['acc'])
        grouped_f1[r['sev']].append(r['f1'])
    agg_acc, agg_f1 = {}, {}
    for sev in sorted(grouped_acc.keys()):
        accs = np.array(grouped_acc[sev])
        f1s = np.array(grouped_f1[sev])
        agg_acc[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
        agg_f1[sev] = f"{f1s.mean():.2f} ± {f1s.std():.2f}"
        label = "clean" if sev == 0 else f"sev{sev}"
        print(f"  {label:<10} acc={agg_acc[sev]}  f1={agg_f1[sev]}")
    print(f"[ED_AGG] method={METHOD_TAG} dataset={dataset} "
          f"clean=\"{agg_acc.get(0,'')}\" "
          f"sev1=\"{agg_acc.get(1,'')}\" sev2=\"{agg_acc.get(2,'')}\" "
          f"sev3=\"{agg_acc.get(3,'')}\" sev4=\"{agg_acc.get(4,'')}\" "
          f"sev5=\"{agg_acc.get(5,'')}\" "
          f"clean_f1=\"{agg_f1.get(0,'')}\" "
          f"sev1_f1=\"{agg_f1.get(1,'')}\" sev2_f1=\"{agg_f1.get(2,'')}\" "
          f"sev3_f1=\"{agg_f1.get(3,'')}\" sev4_f1=\"{agg_f1.get(4,'')}\" "
          f"sev5_f1=\"{agg_f1.get(5,'')}\"")


def main(argv):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[{METHOD_TAG} ED] Using {device}", flush=True)

    data, _ = load_tag(FLAGS.dataset)
    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    y_np = data.y.cpu().numpy()
    num_nodes = int(data.num_nodes)
    clean_edge_index = data.edge_index.clone().to(device)
    num_classes = int(data.y.max().item()) + 1

    print(f"[{METHOD_TAG} ED] {FLAGS.dataset}, N={num_nodes}, "
          f"E={clean_edge_index.size(1)}, C={num_classes}", flush=True)

    splits = build_splits(data)
    all_results = []

    for split_idx, split in enumerate(splits):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_mask_np = split['train'].cpu().numpy().astype(bool)
        test_mask_np = split['test'].cpu().numpy().astype(bool)

        # sev=0 (clean) goes through the same pipeline as perturbed.
        for sev_idx, p in [(0, 0.0)] + SEVERITIES:
            edges = (clean_edge_index if p == 0.0
                     else apply_edge_drop(clean_edge_index, num_nodes, p))
            print(f"  [split {split_idx}] sev={sev_idx} p={p}  "
                  f"E_kept={edges.size(1)}/{clean_edge_index.size(1)}", flush=True)

            emb = learn_node2vec(edges, num_nodes, device)
            acc, macro_f1 = eval_lr(emb, y_np, train_mask_np, test_mask_np)

            all_results.append({
                'split_idx': split_idx,
                'sev': sev_idx,
                'acc': acc,
                'f1': macro_f1,
            })
            print(f"[ED_RAW] method={METHOD_TAG} dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} p={p} "
                  f"test_acc={acc:.4f} macro_f1={macro_f1:.4f}", flush=True)

    aggregate_and_print(all_results, FLAGS.dataset)


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset')
    app.run(main)
