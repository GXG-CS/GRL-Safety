"""
GraphMAE Linear-Probing eval + feature-noise corruption evaluation.

Loads a pretrained GraphMAE model (frozen encoder), trains a logistic
regression head on clean SBERT representations, then re-evaluates the
SAME clean-trained head on representations from noisy features.

Mirrors `sgb/models/ssl/bgrl/eval_feature_noise.py` but uses:
  - DGL (graphmae is DGL-based)
  - PreModel.embed(g, x) to extract encoder output

Outputs structured `[FN_RAW]` and `[FN_AGG]` lines.
"""

import os
import os.path as osp
import sys
import argparse
import collections

import numpy as np
import torch
import dgl

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_GMAE_DIR = osp.dirname(osp.abspath(__file__))               # .../sgb/models/ssl/graphmae
_PROJECT_ROOT = osp.abspath(osp.join(_GMAE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _GMAE_DIR not in sys.path:
    sys.path.insert(0, _GMAE_DIR)

from graphmae.models.edcoder import PreModel  # type: ignore
from sgb.data.tag_registry import load as load_tag  # type: ignore

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# -----------------------------------------------------------------------------
# Feature noise config
# -----------------------------------------------------------------------------
SEVERITIES = [
    (1, 0.1),
    (2, 0.25),
    (3, 0.5),
    (4, 1.0),
    (5, 2.0),
]

DATASET2TASK = {
    "WN18RR": "link", "FB15K237": "link",
    "chemhiv": "graph", "chempcba": "graph",
    "bace": "graph", "bbbp": "graph",
    "cyp450": "graph", "muv": "graph",
    "tox21": "graph", "toxcast": "graph",
}


def apply_feature_noise(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    sigma_rel: float,
    noise_seed: int,
) -> torch.Tensor:
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    if train_mask.device != x.device:
        train_mask = train_mask.to(x.device)
    std = x[train_mask].std(dim=0, keepdim=True)
    g = torch.Generator(device=x.device).manual_seed(int(noise_seed))
    eps = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


def fit_lr_best_c(X_train, y_train, X_val, y_val):
    """Same lbfgs multinomial fit as BGRL FN script."""
    best_acc = -1.0
    best_clf = None
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            solver='lbfgs', multi_class='multinomial',
            C=c, max_iter=500, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best_clf = clf
    return best_clf


def lr_test_acc(clf, X_test, y_test):
    return clf.score(X_test, y_test)


def load_dgl_dataset(name):
    """Load TAG dataset and convert to DGL graph + collected splits.

    Returns:
        graph: DGL graph with ndata 'feat' (clean), 'label'
        splits: list of dicts {train, val, test} of bool tensors, length up to 5
    """
    data, _ = load_tag(name)

    # Materialize features as 768d tensor
    if data.x is not None and data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat[data.x].float()
    elif data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        feat = data.x.float()
    elif hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat.float()
    else:
        raise RuntimeError(f"Cannot extract 768d features for dataset {name}")

    # Squeeze y
    y = data.y.squeeze() if data.y is not None else None

    # Build DGL graph
    src = data.edge_index[0]
    dst = data.edge_index[1]
    graph = dgl.graph((src, dst), num_nodes=feat.size(0))
    graph = graph.remove_self_loop().add_self_loop()
    graph.ndata["feat"] = feat
    if y is not None:
        graph.ndata["label"] = y

    # Build splits (always exactly 5)
    N = feat.size(0)
    n_target = 5
    splits = []
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        for i in range(min(n_target, len(data.train_masks))):
            splits.append({
                'train': data.train_masks[i].bool(),
                'val': data.val_masks[i].bool(),
                'test': data.test_masks[i].bool(),
            })
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        tm = torch.zeros(N, dtype=torch.bool); tm[s['train']] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[s.get('valid', s.get('val'))] = True
        tsm = torch.zeros(N, dtype=torch.bool); tsm[s['test']] = True
        splits.append({'train': tm, 'val': vm, 'test': tsm})
    elif hasattr(data, 'train_mask') and data.train_mask is not None:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            for i in range(min(n_target, tm.size(1))):
                splits.append({
                    'train': tm[:, i].bool(),
                    'val': vm[:, i].bool(),
                    'test': (tsm[:, i] if tsm.dim() == 2 else tsm).bool(),
                })
        else:
            splits.append({
                'train': tm.bool(),
                'val': vm.bool(),
                'test': tsm.bool(),
            })
    else:
        raise RuntimeError(f"No train mask for dataset {name}")
    while len(splits) < n_target:
        splits.append(splits[len(splits) % len(splits)])
    splits = splits[:n_target]

    return graph, splits


def build_graphmae_model(
    num_features: int = 768,
    num_hidden: int = 768,
    num_layers: int = 2,
    num_heads: int = 4,
    mask_rate: float = 0.5,
    replace_rate: float = 0.0,
    loss_fn: str = "sce",
    alpha_l: float = 3.0,
) -> PreModel:
    """Build PreModel matching the joint pretrain config (pretrain_joint.py).

    Must use norm=None + decoder_type='mlp' to match ckpts/graphmae/all/model.pt.
    (Previously used layernorm + gat decoder, which caused state_dict mismatch.)
    """
    return PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_heads,
        activation="prelu",
        feat_drop=0.2,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        norm=None,
        mask_rate=mask_rate,
        encoder_type="gat",
        decoder_type="mlp",
        loss_fn=loss_fn,
        drop_edge_rate=0.0,
        replace_rate=replace_rate,
        alpha_l=alpha_l,
        concat_hidden=False,
    )


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


def run_link_fn(device, model, args):
    """Link (KG edge-type) LP + feature noise eval."""
    data, _ = load_tag(args.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]

    # Build DGL graph for encoder
    src = data.edge_index[0]
    dst = data.edge_index[1]
    graph = dgl.graph((src, dst), num_nodes=data.x.size(0))
    graph = graph.remove_self_loop().add_self_loop()
    graph = graph.to(device)

    feat_clean = data.x.float().to(device)

    labels = data.edge_types
    num_classes = int(labels.unique().shape[0])
    # KG splits: edges ordered as train|val|test
    n_train = len(data.train_idx)
    n_valid = len(data.val_idx)
    n_test = len(data.test_idx)

    all_results = []

    for split_idx in range(5):
        # Clean reps
        with torch.no_grad():
            z = model.embed(graph, feat_clean).detach()
        ei = data.edge_index.to(device)
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
        print(f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}")

        # Use all node features for std (link task)
        all_mask = torch.ones(feat_clean.size(0), dtype=torch.bool, device=device)
        for sev_idx, sigma_rel in SEVERITIES:
            feat_noisy = apply_feature_noise(feat_clean, all_mask, sigma_rel, noise_seed=split_idx*100+sev_idx)
            with torch.no_grad():
                z_noisy = model.embed(graph, feat_noisy).detach()
            edge_z_noisy = ((z_noisy[ei[0]] + z_noisy[ei[1]]) / 2).cpu().numpy()
            edge_z_noisy = normalize(edge_z_noisy, norm='l2')
            noisy_acc = clf.score(edge_z_noisy[n_train+n_valid:], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "sigma_rel": sigma_rel, "test_acc": noisy_acc})
            print(f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} test_acc={noisy_acc:.4f}")

    _print_agg(all_results, args.dataset, "GraphMAE_LP", "FN")


def run_graph_fn(device, model, args):
    """Graph classification LP + feature noise eval."""
    # Load TAG per-graph data
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", args.dataset, "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    n_graphs = slices["y"].shape[0] - 1

    # Build per-graph DGL graphs
    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        feat = node_text_feat[atom_idx].float()
        ei = merged.edge_index[:, es:ee]
        g = dgl.graph((ei[0], ei[1]), num_nodes=feat.size(0))
        g = g.remove_self_loop().add_self_loop()
        g.ndata["feat"] = feat
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
        embeds = []
        batch_size = 512
        for start in range(0, len(graph_list), batch_size):
            batch_graphs = graph_list[start:start+batch_size]
            bg = dgl.batch(batch_graphs).to(device)
            feat = bg.ndata["feat"].to(device)
            with torch.no_grad():
                z = model.embed(bg, feat).detach()
            # Mean pool per graph
            bg.ndata['h'] = z
            g_emb = dgl.readout_nodes(bg, 'h', op='mean')
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
        print(f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}")

        # Compute global std from training graphs
        train_graphs = [graphs[i] for i in train_idx]
        all_train_feat = torch.cat([g.ndata["feat"] for g in train_graphs], dim=0)
        feat_std = all_train_feat.std(dim=0, keepdim=True)

        for sev_idx, sigma_rel in SEVERITIES:
            g_gen = torch.Generator().manual_seed(int(split_idx * 100 + sev_idx))
            noisy_graphs = []
            for g in graphs:
                gc = dgl.graph(g.edges(), num_nodes=g.num_nodes())
                gc = gc.remove_self_loop().add_self_loop()
                eps = torch.randn(g.ndata["feat"].shape, generator=g_gen)
                gc.ndata["feat"] = g.ndata["feat"] + sigma_rel * feat_std * eps
                noisy_graphs.append(gc)
            noisy_emb = normalize(get_graph_embeds(noisy_graphs), norm='l2')
            noisy_acc = clf.score(noisy_emb[test_idx], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "sigma_rel": sigma_rel, "test_acc": noisy_acc})
            print(f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} test_acc={noisy_acc:.4f}")

    _print_agg(all_results, args.dataset, "GraphMAE_LP", "FN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True, help="Path to pretrain model.pt")
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[GraphMAE FN] Using {device}")

    task = DATASET2TASK.get(args.dataset, "node")

    if task in ("link", "graph"):
        # Build model and load state_dict (shared for link/graph)
        model = build_graphmae_model(num_features=768)
        state = torch.load(args.ckpt_path, map_location=device)
        if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        model = model.to(device).eval()
        print(f"[GraphMAE FN] Loaded model, task={task}")

        if task == "link":
            run_link_fn(device, model, args)
        else:
            run_graph_fn(device, model, args)
        return

    # ====================================================================
    # Node classification (original logic, untouched)
    # ====================================================================

    # Load DGL dataset and splits
    graph, splits = load_dgl_dataset(args.dataset)
    num_classes = int(graph.ndata["label"].max().item()) + 1
    print(f"[GraphMAE FN] Dataset: {args.dataset}, num_nodes={graph.num_nodes()}, "
          f"num_classes={num_classes}, splits={len(splits)}")

    graph = graph.to(device)
    feat_clean = graph.ndata["feat"].to(device)
    y_int = graph.ndata["label"].cpu().numpy().astype(np.int64)

    # Build model and load state_dict
    model = build_graphmae_model(num_features=feat_clean.size(1))
    state = torch.load(args.ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[GraphMAE FN] Loaded model from {args.ckpt_path}")

    all_results = []

    for split_idx, split in enumerate(splits):
        train_mask = split['train'].to(device)
        val_mask = split['val'].to(device)
        test_mask = split['test'].to(device)

        train_np = train_mask.cpu().numpy()
        val_np = val_mask.cpu().numpy()
        test_np = test_mask.cpu().numpy()

        # Clean embeddings
        with torch.no_grad():
            clean_emb = model.embed(graph, feat_clean).cpu().numpy()
        clean_emb = normalize(clean_emb, norm='l2')

        X_train, y_train = clean_emb[train_np], y_int[train_np]
        X_val, y_val = clean_emb[val_np], y_int[val_np]
        X_test_clean, y_test = clean_emb[test_np], y_int[test_np]

        clf = fit_lr_best_c(X_train, y_train, X_val, y_val)

        clean_acc = lr_test_acc(clf, X_test_clean, y_test) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_acc})
        print(
            f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
            f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f}"
        )

        for sev_idx, sigma_rel in SEVERITIES:
            feat_noisy = apply_feature_noise(
                feat_clean, train_mask, sigma_rel, noise_seed=split_idx * 100 + sev_idx
            )
            with torch.no_grad():
                noisy_emb = model.embed(graph, feat_noisy).cpu().numpy()
            noisy_emb = normalize(noisy_emb, norm='l2')
            X_test_noisy = noisy_emb[test_np]
            noise_acc = lr_test_acc(clf, X_test_noisy, y_test) * 100.0
            all_results.append({
                "split_idx": split_idx, "sev": sev_idx,
                "sigma_rel": sigma_rel, "test_acc": noise_acc,
            })
            print(
                f"[FN_RAW] method=GraphMAE_LP dataset={args.dataset} "
                f"split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} "
                f"test_acc={noise_acc:.4f}"
            )

    # Aggregate
    print("\n=== GraphMAE Feature Noise Results (aggregated over splits) ===")
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
        f"[FN_AGG] method=GraphMAE_LP dataset={args.dataset} "
        f"clean=\"{aggregated.get(0, '')}\" "
        f"sev1=\"{aggregated.get(1, '')}\" "
        f"sev2=\"{aggregated.get(2, '')}\" "
        f"sev3=\"{aggregated.get(3, '')}\" "
        f"sev4=\"{aggregated.get(4, '')}\" "
        f"sev5=\"{aggregated.get(5, '')}\""
    )


if __name__ == "__main__":
    main()
