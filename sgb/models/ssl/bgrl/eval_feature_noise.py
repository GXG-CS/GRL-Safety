"""
BGRL Linear-Probing eval + feature-noise corruption evaluation.

Loads a pretrained BGRL encoder (frozen), trains a logistic regression head
on clean SBERT representations, then re-evaluates the SAME clean-trained head
on representations produced from noisy features (5 severity levels).

The classifier head is NOT re-fitted on noisy features — per spec.

Outputs structured `[FN_RAW]` and `[FN_AGG]` lines for downstream aggregation.
"""

import os
import os.path as osp
import sys
import logging
import collections

import numpy as np
import torch
from absl import app, flags

# -----------------------------------------------------------------------------
# Path setup: bgrl repo + project root
# -----------------------------------------------------------------------------
_BGRL_DIR = osp.dirname(osp.abspath(__file__))           # .../sgb/models/ssl/bgrl
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)

from bgrl import GCN  # type: ignore
from sgb.data.tag_registry import load as load_tag  # type: ignore

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize
from sklearn import metrics

# -----------------------------------------------------------------------------
# CLI flags
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'Target dataset name (loaded via tag_registry).')
flags.DEFINE_string('ckpt_path', None, 'Path to pretrained BGRL encoder.pt.')
flags.DEFINE_multi_integer('graph_encoder_layer', None,
                           'Encoder layer sizes after input layer (default [768, 768]).')

# -----------------------------------------------------------------------------
# Feature noise config (per spec)
# -----------------------------------------------------------------------------
SEVERITIES = [
    (1, 0.1),
    (2, 0.25),
    (3, 0.5),
    (4, 1.0),
    (5, 2.0),
]


def apply_feature_noise(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    sigma_rel: float,
    noise_seed: int,
) -> torch.Tensor:
    """Additive relative Gaussian noise, scaled by per-dim std of training nodes."""
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    if train_mask.device != x.device:
        train_mask = train_mask.to(x.device)
    std = x[train_mask].std(dim=0, keepdim=True)
    g = torch.Generator(device=x.device).manual_seed(int(noise_seed))
    eps = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


def fit_lr_best_c(X_train, y_train, X_val, y_val):
    """Fit logistic regression with a small C grid; return best classifier (by val acc).

    Uses lbfgs (multinomial) instead of liblinear (OvR + dual coordinate descent),
    which is dramatically faster on high-dimensional multi-class problems
    (e.g. arxiv: 169K samples × 768d × 40 classes).
    """
    # Convert one-hot back to integer labels for multinomial lbfgs
    y_train_int = np.argmax(y_train, axis=1)
    y_val_int = np.argmax(y_val, axis=1)

    best_acc = -1.0
    best_clf = None
    # 5-point grid covers 4 orders of magnitude — enough for robust selection
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            solver='lbfgs', multi_class='multinomial',
            C=c, max_iter=500, n_jobs=-1,
        )
        clf.fit(X_train, y_train_int)
        acc = clf.score(X_val, y_val_int)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
    return best_clf


def lr_test_acc(clf, X_test, y_test):
    """y_test is one-hot bool; classifier's .predict returns integer labels."""
    y_test_int = np.argmax(y_test, axis=1)
    return clf.score(X_test, y_test_int)


def compute_reps(encoder, data):
    encoder.eval()
    with torch.no_grad():
        reps = encoder(data)
    return reps


def _idx_to_mask(idx, num_nodes):
    """Convert index tensor to boolean mask."""
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = True
    return mask


def build_splits(data, n_target=5):
    """Return list of dicts {train, val, test}, exactly n_target entries."""
    splits = []
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        n_avail = len(data.train_masks)
        for i in range(min(n_target, n_avail)):
            splits.append({
                'train': data.train_masks[i],
                'val': data.val_masks[i],
                'test': data.test_masks[i],
            })
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        N = data.num_nodes
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
        splits.append({'train': tm, 'val': vm, 'test': tsm})
    else:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            for i in range(min(n_target, tm.size(1))):
                splits.append({
                    'train': tm[:, i],
                    'val': vm[:, i],
                    'test': tsm[:, i] if tsm.dim() == 2 else tsm,
                })
        else:
            splits.append({'train': tm, 'val': vm, 'test': tsm})
    # Pad to n_target by repeating (different noise seeds per split index)
    while len(splits) < n_target:
        splits.append(splits[len(splits) % len(splits)])
    return splits[:n_target]


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[BGRL FN] Using {device}")

    task = DATASET2TASK.get(FLAGS.dataset, "node")

    if task in ("link", "graph"):
        # --- Encoder (shared for link/graph) ---
        layers = list(FLAGS.graph_encoder_layer) if FLAGS.graph_encoder_layer else [768, 768]
        # For link/graph we need to load a dummy graph first to get input_dim
        # Actually, all TAG datasets use 768d SBERT features
        input_size = 768
        encoder = GCN([input_size] + layers, batchnorm=True)
        ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            encoder.load_state_dict(ckpt['model'])
        else:
            encoder.load_state_dict(ckpt)
        encoder = encoder.to(device).eval()
        print(f"[BGRL FN] Loaded encoder, layers=[{input_size}]+{layers}, task={task}")

        if task == "link":
            run_link_fn(device, encoder)
        else:
            run_graph_fn(device, encoder)
        return

    # ====================================================================
    # Node classification (original logic, untouched)
    # ====================================================================

    # ---------------- Data ----------------
    data, _ = load_tag(FLAGS.dataset)

    # Materialize x = SBERT 768d feature
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat

    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    data = data.to(device)
    print(f"[BGRL FN] Dataset: {FLAGS.dataset}, x.shape={tuple(data.x.shape)}, "
          f"num_nodes={data.x.size(0)}, num_classes={int(data.y.max().item()) + 1}")

    # ---------------- Encoder ----------------
    layers = list(FLAGS.graph_encoder_layer) if FLAGS.graph_encoder_layer else [768, 768]
    input_size = data.x.size(1)
    encoder = GCN([input_size] + layers, batchnorm=True)

    ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        encoder.load_state_dict(ckpt['model'])
    else:
        encoder.load_state_dict(ckpt)
    encoder = encoder.to(device).eval()
    print(f"[BGRL FN] Loaded encoder from {FLAGS.ckpt_path}, layers=[{input_size}]+{layers}")

    # ---------------- Splits ----------------
    splits = build_splits(data)
    print(f"[BGRL FN] Built {len(splits)} split(s)")

    # ---------------- LR + noise loop ----------------
    x_clean = data.x.clone()

    # One-hot encoder for labels (fit once)
    y_np = data.y.cpu().numpy()
    ohe = OneHotEncoder(categories='auto', sparse_output=False)
    y_oh = ohe.fit_transform(y_np.reshape(-1, 1)).astype(bool)

    all_results = []

    for split_idx, split in enumerate(splits):
        train_mask = split['train']
        val_mask = split['val']
        test_mask = split['test']
        if not isinstance(train_mask, torch.Tensor):
            train_mask = torch.tensor(train_mask)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device) if isinstance(val_mask, torch.Tensor) else torch.tensor(val_mask).to(device)
        test_mask = test_mask.to(device) if isinstance(test_mask, torch.Tensor) else torch.tensor(test_mask).to(device)

        train_np = train_mask.cpu().numpy().astype(bool)
        val_np = val_mask.cpu().numpy().astype(bool)
        test_np = test_mask.cpu().numpy().astype(bool)

        # Clean reps + LR fit
        data.x = x_clean
        clean_reps = compute_reps(encoder, data).cpu().numpy()
        clean_reps = normalize(clean_reps, norm='l2')

        X_train = clean_reps[train_np]
        y_train = y_oh[train_np]
        X_val = clean_reps[val_np]
        y_val = y_oh[val_np]
        X_test_clean = clean_reps[test_np]
        y_test = y_oh[test_np]

        clf = fit_lr_best_c(X_train, y_train, X_val, y_val)

        clean_acc = lr_test_acc(clf, X_test_clean, y_test) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_acc})
        print(
            f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
            f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}"
        )

        for sev_idx, sigma_rel in SEVERITIES:
            data.x = apply_feature_noise(
                x_clean, train_mask, sigma_rel, noise_seed=split_idx * 100 + sev_idx
            )
            noisy_reps = compute_reps(encoder, data).cpu().numpy()
            noisy_reps = normalize(noisy_reps, norm='l2')
            X_test_noisy = noisy_reps[test_np]
            noise_acc = lr_test_acc(clf, X_test_noisy, y_test) * 100.0
            all_results.append({
                "split_idx": split_idx, "sev": sev_idx,
                "sigma_rel": sigma_rel, "test_acc": noise_acc,
            })
            print(
                f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
                f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} "
                f"test_acc={noise_acc:.4f}"
            )

        data.x = x_clean  # restore for next split

    # ---------------- Aggregation + output ----------------
    print("\n=== BGRL Feature Noise Results (aggregated over splits) ===")
    grouped = collections.defaultdict(list)
    for row in all_results:
        grouped[row["sev"]].append(row["test_acc"])

    label_for_sev = {0: "clean   "}
    for sev_idx, sigma_rel in SEVERITIES:
        label_for_sev[sev_idx] = f"sev{sev_idx} σ={sigma_rel}"

    aggregated = {}
    for sev in sorted(grouped.keys()):
        accs = np.array(grouped[sev], dtype=np.float64)
        mean, std = accs.mean(), accs.std()
        aggregated[sev] = f"{mean:.2f} ± {std:.2f}"
        print(f"  {label_for_sev[sev]:<14}  {aggregated[sev]}")

    print(
        f"[FN_AGG] method=BGRL_LP dataset={FLAGS.dataset} "
        f"clean=\"{aggregated.get(0, '')}\" "
        f"sev1=\"{aggregated.get(1, '')}\" "
        f"sev2=\"{aggregated.get(2, '')}\" "
        f"sev3=\"{aggregated.get(3, '')}\" "
        f"sev4=\"{aggregated.get(4, '')}\" "
        f"sev5=\"{aggregated.get(5, '')}\""
    )


DATASET2TASK = {
    "WN18RR": "link", "FB15K237": "link",
    "chemhiv": "graph", "chempcba": "graph",
    "bace": "graph", "bbbp": "graph",
    "cyp450": "graph", "muv": "graph",
    "tox21": "graph", "toxcast": "graph",
}


def run_link_fn(device, encoder):
    """Link (KG edge-type) LP + feature noise eval."""
    data, _ = load_tag(FLAGS.dataset)
    # Materialize 768d features
    if hasattr(data, 'node_text_feat') and data.node_text_feat is not None:
        if data.x is None or (data.x.ndim == 2 and data.x.size(1) != 768):
            data.x = data.node_text_feat
        elif data.x.dtype == torch.long and data.x.ndim == 1:
            data.x = data.node_text_feat[data.x]
    data = data.to(device)

    labels = data.edge_types
    num_classes = int(labels.unique().shape[0])
    # KG splits: edges ordered as train|val|test
    n_train = len(data.train_idx)
    n_valid = len(data.val_idx)
    n_test = len(data.test_idx)

    x_clean = data.x.clone()
    all_results = []

    for split_idx in range(5):
        # Clean reps
        data.x = x_clean
        z = compute_reps(encoder, data).detach()
        ei = data.edge_index
        edge_z = ((z[ei[0]] + z[ei[1]]) / 2).cpu().numpy()
        edge_z = normalize(edge_z, norm='l2')
        y_np = labels.cpu().numpy()

        X_tr, y_tr = edge_z[:n_train], y_np[:n_train]
        X_va, y_va = edge_z[n_train:n_train+n_valid], y_np[n_train:n_train+n_valid]
        X_te, y_te = edge_z[n_train+n_valid:], y_np[n_train+n_valid:]

        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1.0, max_iter=500, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        clean_acc = clf.score(X_te, y_te) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_acc})
        print(f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}")

        # Use all node features for std (link task)
        all_mask = torch.ones(x_clean.size(0), dtype=torch.bool, device=device)
        for sev_idx, sigma_rel in SEVERITIES:
            data.x = apply_feature_noise(x_clean, all_mask, sigma_rel, noise_seed=split_idx*100+sev_idx)
            z_noisy = compute_reps(encoder, data).detach()
            edge_z_noisy = ((z_noisy[ei[0]] + z_noisy[ei[1]]) / 2).cpu().numpy()
            edge_z_noisy = normalize(edge_z_noisy, norm='l2')
            noisy_acc = clf.score(edge_z_noisy[n_train+n_valid:], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "sigma_rel": sigma_rel, "test_acc": noisy_acc})
            print(f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} test_acc={noisy_acc:.4f}")
        data.x = x_clean

    _print_agg(all_results, FLAGS.dataset, "BGRL_LP", "FN")


def run_graph_fn(device, encoder):
    """Graph classification LP + feature noise eval."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Data
    from torch_scatter import scatter_mean

    # Load TAG per-graph data
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", FLAGS.dataset, "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    n_graphs = slices["y"].shape[0] - 1
    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        g = Data(x=node_text_feat[atom_idx], edge_index=merged.edge_index[:, es:ee],
                 y=merged.y[slices["y"][i]:slices["y"][i+1]])
        graphs.append(g)

    labels = merged.y.squeeze().cpu().numpy()

    # Random 80/10/10 split (seeded, no OGB dependency)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr = int(0.8 * n_graphs)
    n_va = int(0.1 * n_graphs)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr+n_va]
    test_idx = perm[n_tr+n_va:]

    def get_graph_embeds(graph_list):
        loader = PyGDataLoader(graph_list, batch_size=512, shuffle=False, num_workers=4)
        embeds = []
        for batch in loader:
            batch = batch.to(device)
            z = compute_reps(encoder, batch).detach()
            g_emb = scatter_mean(z, batch.batch, dim=0)
            embeds.append(g_emb.cpu())
        return torch.cat(embeds, dim=0).numpy()

    all_results = []
    for split_idx in range(5):
        # Clean
        all_emb = normalize(get_graph_embeds(graphs), norm='l2')
        X_tr, X_va, X_te = all_emb[train_idx], all_emb[val_idx], all_emb[test_idx]
        y_tr, y_va, y_te = labels[train_idx], labels[val_idx], labels[test_idx]
        clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=500, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        clean_acc = clf.score(X_te, y_te) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_acc})
        print(f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}")

        # Compute global std from training graphs
        train_graphs = [graphs[i] for i in train_idx]
        all_train_feat = torch.cat([g.x for g in train_graphs], dim=0)
        feat_std = all_train_feat.std(dim=0, keepdim=True)

        for sev_idx, sigma_rel in SEVERITIES:
            g_gen = torch.Generator().manual_seed(int(split_idx * 100 + sev_idx))
            noisy_graphs = []
            for g in graphs:
                gc = g.clone()
                eps = torch.randn(gc.x.shape, generator=g_gen)
                gc.x = gc.x + sigma_rel * feat_std * eps
                noisy_graphs.append(gc)
            noisy_emb = normalize(get_graph_embeds(noisy_graphs), norm='l2')
            noisy_acc = clf.score(noisy_emb[test_idx], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "sigma_rel": sigma_rel, "test_acc": noisy_acc})
            print(f"[FN_RAW] method=BGRL_LP dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} test_acc={noisy_acc:.4f}")

    _print_agg(all_results, FLAGS.dataset, "BGRL_LP", "FN")


def _print_agg(all_results, dataset, method, corruption_type):
    """Print aggregated results in standard format."""
    grouped = collections.defaultdict(list)
    for row in all_results:
        grouped[row["sev"]].append(row["test_acc"])
    aggregated = {}
    for sev in sorted(grouped.keys()):
        accs = np.array(grouped[sev], dtype=np.float64)
        aggregated[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
    tag = "FN" if corruption_type == "FN" else "ED"
    print(
        f"[{tag}_AGG] method={method} dataset={dataset} "
        f"clean=\"{aggregated.get(0, '')}\" "
        f"sev1=\"{aggregated.get(1, '')}\" "
        f"sev2=\"{aggregated.get(2, '')}\" "
        f"sev3=\"{aggregated.get(3, '')}\" "
        f"sev4=\"{aggregated.get(4, '')}\" "
        f"sev5=\"{aggregated.get(5, '')}\""
    )


if __name__ == "__main__":
    app.run(main)
