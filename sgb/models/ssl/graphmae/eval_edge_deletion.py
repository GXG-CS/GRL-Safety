"""
GraphMAE Linear-Probing eval + edge-deletion corruption evaluation.

Loads a GraphMAE joint-pretrained PreModel (frozen encoder), trains a
logistic regression head on clean SBERT representations, then re-evaluates
the SAME clean-trained head on representations produced from graphs with
randomly dropped edges (5 severity levels: p in 0.05/0.10/0.20/0.30/0.50).

The classifier head is NOT re-fitted on corrupted graphs — per spec.
Node features are unchanged; only the edge set is perturbed. Self-loops
are stripped from the base edge set before dropping, then re-added to the
DGL graph each severity so the encoder always sees a self-looped graph.

Model builder mirrors `pretrain_joint.py` / `eval_joint.py` exactly
(encoder=gat, decoder=mlp, norm=None, drop_edge_rate=0.0). This is
intentionally NOT the same config as the sibling `eval_feature_noise.py`,
whose builder uses layernorm + gat decoder and therefore would not match
the `ckpts/graphmae/all/model.pt` joint checkpoint.

Outputs structured `[ED_RAW]` and `[ED_AGG]` lines for downstream
aggregation, matching the format of `sgb/models/ssl/bgrl/eval_edge_deletion.py`.
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
# Path setup: graphmae repo + project root
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
# Edge deletion config (per spec; identical to BGRL eval_edge_deletion.py)
# -----------------------------------------------------------------------------
SEVERITIES = [
    (1, 0.05),
    (2, 0.10),
    (3, 0.20),
    (4, 0.30),
    (5, 0.50),
]


def apply_edge_drop(
    edge_index: torch.Tensor,
    num_nodes: int,
    p: float,
) -> torch.Tensor:
    """Random Bernoulli edge drop with undirected-safe semantics.

    - (u,v) and (v,u) are treated as one undirected unit (canonical key on
      (min, max)), so both directions drop or stay together.
    - Self-loops are preserved explicitly (though the caller strips them
      before calling — this is a belt-and-braces guard).
    - Uses torch's default RNG; reproducibility comes from the global
      torch.manual_seed set at script entry.
    """
    if p <= 0.0:
        return edge_index
    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    key = u.long() * num_nodes + v.long()
    _, inverse = torch.unique(key, return_inverse=True)
    num_undirected = int(inverse.max().item()) + 1
    keep_per_undirected = (
        torch.rand(num_undirected, device=edge_index.device) >= p
    )
    keep = keep_per_undirected[inverse]
    keep = keep | (src == dst)  # preserve self-loops
    return edge_index[:, keep]


# -----------------------------------------------------------------------------
# Model builder — must mirror pretrain_joint.py's PreModel args exactly
# (see also eval_joint.py:build_joint_model)
# -----------------------------------------------------------------------------
def build_joint_model(
    num_features: int = 768,
    num_hidden: int = 768,
    num_layers: int = 2,
    num_heads: int = 4,
    mask_rate: float = 0.5,
    replace_rate: float = 0.0,
    loss_fn: str = "sce",
    alpha_l: float = 3.0,
) -> PreModel:
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


# -----------------------------------------------------------------------------
# Data utilities (tag_registry -> DGL)
# -----------------------------------------------------------------------------
def _materialize_feat(data) -> torch.Tensor:
    """Resolve the 768d SBERT feature tensor regardless of cache layout."""
    if data.x is not None and data.x.dtype == torch.long and data.x.ndim == 1:
        return data.node_text_feat[data.x].float()
    if data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        return data.x.float()
    if hasattr(data, "node_text_feat") and data.node_text_feat is not None:
        return data.node_text_feat.float()
    raise RuntimeError("Cannot materialize 768d feature tensor")


def _extract_splits(data, n_target: int = 5):
    """Return exactly `n_target` dicts of bool train/val/test masks."""
    import torch
    N = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
    splits = []
    if hasattr(data, "train_masks") and data.train_masks is not None:
        for i in range(min(n_target, len(data.train_masks))):
            splits.append({
                "train": data.train_masks[i].bool(),
                "val": data.val_masks[i].bool(),
                "test": data.test_masks[i].bool(),
            })
    elif hasattr(data, "splits") and isinstance(data.splits, dict):
        s = data.splits
        tm = torch.zeros(N, dtype=torch.bool); tm[s['train']] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[s.get('valid', s.get('val'))] = True
        tsm = torch.zeros(N, dtype=torch.bool); tsm[s['test']] = True
        splits.append({"train": tm, "val": vm, "test": tsm})
    elif hasattr(data, "train_mask") and data.train_mask is not None:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            for i in range(min(n_target, tm.size(1))):
                splits.append({
                    "train": tm[:, i].bool(),
                    "val": vm[:, i].bool(),
                    "test": (tsm[:, i] if tsm.dim() == 2 else tsm).bool(),
                })
        else:
            splits.append({
                "train": tm.bool(),
                "val": vm.bool(),
                "test": tsm.bool(),
            })
    else:
        raise RuntimeError("Dataset has no train mask")
    while len(splits) < n_target:
        splits.append(splits[len(splits) % len(splits)])
    return splits[:n_target]


def load_tag_dataset(name: str):
    """Load dataset via tag_registry and return pieces needed for ED eval.

    Returns:
        feat: (N, 768) float tensor on CPU
        y:    (N,) long tensor on CPU
        base_edge_index: (2, E) long tensor on CPU with self-loops stripped
        num_nodes: int
        splits: list of {train, val, test} bool-mask dicts
    """
    data, _ = load_tag(name)
    feat = _materialize_feat(data)
    y = data.y.squeeze() if data.y is not None and data.y.dim() > 1 else data.y

    ei = data.edge_index.long()
    src, dst = ei[0], ei[1]
    non_self = src != dst
    base_ei = ei[:, non_self]

    splits = _extract_splits(data)
    return feat, y.long(), base_ei, int(feat.size(0)), splits


def build_dgl_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    feat: torch.Tensor,
    y_device: torch.Tensor,
) -> dgl.DGLGraph:
    """Fresh DGL graph with self-loops re-added + ndata attached.

    Caller is responsible for passing `edge_index` already on the target
    device. Graph is constructed on the same device as the edge_index.
    """
    g = dgl.graph(
        (edge_index[0], edge_index[1]),
        num_nodes=num_nodes,
    )
    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = feat
    g.ndata["label"] = y_device
    return g


# -----------------------------------------------------------------------------
# Linear probe helpers
# -----------------------------------------------------------------------------
def fit_lr_best_c(X_train, y_train, X_val, y_val):
    """Fit logistic regression with a small C grid; return best by val acc."""
    best_acc, best_clf = -1.0, None
    for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(
            solver="lbfgs", multi_class="multinomial",
            C=c, max_iter=500, n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
        if acc > best_acc:
            best_acc, best_clf = acc, clf
    return best_clf


DATASET2TASK = {
    "WN18RR": "link", "FB15K237": "link",
    "chemhiv": "graph", "chempcba": "graph",
    "bace": "graph", "bbbp": "graph",
    "cyp450": "graph", "muv": "graph",
    "tox21": "graph", "toxcast": "graph",
}


def _print_agg(all_results, dataset, method, corruption_type):
    grouped = collections.defaultdict(list)
    for row in all_results:
        grouped[row["sev"]].append(row["test_acc"])
    aggregated = {}
    for sev in sorted(grouped.keys()):
        accs = np.array(grouped[sev], dtype=np.float64)
        aggregated[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
    tag = corruption_type
    print(
        f"[{tag}_AGG] method={method} dataset={dataset} "
        f"clean=\"{aggregated.get(0, '')}\" "
        f"sev1=\"{aggregated.get(1, '')}\" sev2=\"{aggregated.get(2, '')}\" "
        f"sev3=\"{aggregated.get(3, '')}\" sev4=\"{aggregated.get(4, '')}\" "
        f"sev5=\"{aggregated.get(5, '')}\""
    )


def run_link_ed(device, model, args):
    """Link (KG edge-type) LP + edge deletion eval using DGL + GraphMAE."""
    data, _ = load_tag(args.dataset)
    feat = _materialize_feat(data).to(device)
    ei = data.edge_index.long()

    labels = data.edge_types
    n_train = len(data.train_idx)
    n_valid = len(data.val_idx)

    num_nodes = int(feat.size(0))
    clean_ei = ei.to(device)
    all_results = []

    for split_idx in range(5):
        # Clean: build DGL graph, encode, classify edges
        g_clean = dgl.graph((clean_ei[0], clean_ei[1]), num_nodes=num_nodes)
        g_clean = g_clean.remove_self_loop().add_self_loop()
        with torch.no_grad():
            z = model.embed(g_clean.to(device), feat).detach()
        full_ei = clean_ei
        edge_z = ((z[full_ei[0]] + z[full_ei[1]]) / 2).cpu().numpy()
        edge_z = normalize(edge_z, norm='l2')
        y_np = labels.cpu().numpy()

        X_tr, y_tr = edge_z[:n_train], y_np[:n_train]
        X_va, y_va = edge_z[n_train:n_train+n_valid], y_np[n_train:n_train+n_valid]
        X_te, y_te = edge_z[n_train+n_valid:], y_np[n_train+n_valid:]

        clf = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                                 C=1.0, max_iter=500, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        clean_acc = clf.score(X_te, y_te) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "p": 0.0, "test_acc": clean_acc})
        print(f"[ED_RAW] method=GraphMAE_LP dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 p=0.0 test_acc={clean_acc:.4f}")

        # Only drop train edges, keep val/test intact
        train_ei = clean_ei[:, :n_train]
        rest_ei = clean_ei[:, n_train:]
        for sev_idx, p in SEVERITIES:
            dropped_ei = apply_edge_drop(train_ei, num_nodes, p=p)
            n_tr_new = dropped_ei.size(1)
            corrupted_ei = torch.cat([dropped_ei, rest_ei], dim=1)
            g_noisy = dgl.graph((corrupted_ei[0], corrupted_ei[1]), num_nodes=num_nodes)
            g_noisy = g_noisy.remove_self_loop().add_self_loop()
            with torch.no_grad():
                z_noisy = model.embed(g_noisy.to(device), feat).detach()
            edge_z_noisy = ((z_noisy[corrupted_ei[0]] + z_noisy[corrupted_ei[1]]) / 2).cpu().numpy()
            edge_z_noisy = normalize(edge_z_noisy, norm='l2')
            noisy_acc = clf.score(edge_z_noisy[n_tr_new + n_valid:], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "p": p, "test_acc": noisy_acc})
            print(f"[ED_RAW] method=GraphMAE_LP dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} p={p} test_acc={noisy_acc:.4f}")

    _print_agg(all_results, args.dataset, "GraphMAE_LP", "ED")


def run_graph_ed(device, model, args):
    """Graph classification LP + edge deletion eval using DGL + GraphMAE."""
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", args.dataset,
                       "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    n_graphs = slices["y"].shape[0] - 1

    # Build list of (feat_tensor, edge_index_tensor, y) per graph
    graph_feats = []
    graph_edges = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        feat_i = node_text_feat[atom_idx].float()
        ei_i = merged.edge_index[:, es:ee]
        graph_feats.append(feat_i)
        graph_edges.append(ei_i)

    labels = merged.y.squeeze().cpu().numpy()

    # Random 80/10/10 split (seeded, no OGB dependency)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr = int(0.8 * n_graphs)
    n_va = int(0.1 * n_graphs)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr+n_va]
    test_idx = perm[n_tr+n_va:]

    def get_graph_embeds(feat_list, edge_list):
        """Encode graphs in batches using DGL batching, mean-pool to graph-level."""
        batch_size = 512
        embeds = []
        for start in range(0, len(feat_list), batch_size):
            end = min(start + batch_size, len(feat_list))
            dgl_graphs = []
            feats_batch = []
            for j in range(start, end):
                n_nodes = feat_list[j].size(0)
                g = dgl.graph((edge_list[j][0], edge_list[j][1]), num_nodes=n_nodes)
                g = g.remove_self_loop().add_self_loop()
                dgl_graphs.append(g)
                feats_batch.append(feat_list[j])
            bg = dgl.batch(dgl_graphs).to(device)
            cat_feat = torch.cat(feats_batch, dim=0).to(device)
            with torch.no_grad():
                z = model.embed(bg, cat_feat).detach()
            # Mean pool per graph
            num_nodes_list = bg.batch_num_nodes()
            splits = torch.cumsum(num_nodes_list, dim=0).cpu().tolist()
            splits = [0] + splits
            for k in range(len(dgl_graphs)):
                g_emb = z[splits[k]:splits[k+1]].mean(dim=0)
                embeds.append(g_emb.cpu())
        return torch.stack(embeds, dim=0).numpy()

    all_results = []
    for split_idx in range(5):
        all_emb = normalize(get_graph_embeds(graph_feats, graph_edges), norm='l2')
        X_tr, X_te = all_emb[train_idx], all_emb[test_idx]
        y_tr, y_te = labels[train_idx], labels[test_idx]
        clf = LogisticRegression(solver='lbfgs', C=1.0, max_iter=500, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        clean_acc = clf.score(X_te, y_te) * 100.0
        all_results.append({"split_idx": split_idx, "sev": 0, "p": 0.0, "test_acc": clean_acc})
        print(f"[ED_RAW] method=GraphMAE_LP dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 p=0.0 test_acc={clean_acc:.4f}")

        for sev_idx, p in SEVERITIES:
            noisy_edges = []
            for gi in range(len(graph_edges)):
                ei_orig = graph_edges[gi]
                n_nodes = graph_feats[gi].size(0)
                ei_dropped = apply_edge_drop(ei_orig, n_nodes, p=p)
                noisy_edges.append(ei_dropped)
            noisy_emb = normalize(get_graph_embeds(graph_feats, noisy_edges), norm='l2')
            noisy_acc = clf.score(noisy_emb[test_idx], y_te) * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "p": p, "test_acc": noisy_acc})
            print(f"[ED_RAW] method=GraphMAE_LP dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} p={p} test_acc={noisy_acc:.4f}")

    _print_agg(all_results, args.dataset, "GraphMAE_LP", "ED")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--ckpt_path", required=True,
        help="Path to graphmae joint pretrain model.pt",
    )
    args = parser.parse_args()

    # One global seed — apply_edge_drop uses the global RNG so the sequence
    # of drops across (split, severity) is reproducible.
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[GraphMAE ED] Using {device}")

    task = DATASET2TASK.get(args.dataset, "node")

    if task in ("link", "graph"):
        # Build model for link / graph tasks (768d input, no dataset-specific loading)
        model = build_joint_model(num_features=768)
        state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state)
        model = model.to(device).eval()
        print(f"[GraphMAE ED] Loaded encoder, task={task}")
        if task == "link":
            run_link_ed(device, model, args)
        else:
            run_graph_ed(device, model, args)
        return

    # ====================================================================
    # Node classification (original logic, untouched)
    # ====================================================================

    feat_cpu, y_cpu, base_ei, num_nodes, splits = load_tag_dataset(args.dataset)
    num_classes = int(y_cpu.max().item()) + 1
    print(
        f"[GraphMAE ED] Dataset: {args.dataset}, num_nodes={num_nodes}, "
        f"num_edges(no-self)={base_ei.size(1)}, num_classes={num_classes}, "
        f"splits={len(splits)}"
    )

    feat = feat_cpu.to(device)
    y_device = y_cpu.to(device)
    y_int = y_cpu.cpu().numpy().astype(np.int64)
    base_ei_dev = base_ei.to(device)

    # Model
    model = build_joint_model(num_features=feat.size(1))
    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[GraphMAE ED] Loaded joint ckpt from {args.ckpt_path}")

    # Clean graph — built once, reused across all splits
    clean_graph = build_dgl_graph(base_ei_dev, num_nodes, feat, y_device)

    all_results = []

    for split_idx, split in enumerate(splits):
        train_np = split["train"].cpu().numpy().astype(bool)
        val_np = split["val"].cpu().numpy().astype(bool)
        test_np = split["test"].cpu().numpy().astype(bool)

        # Clean reps + LR fit
        with torch.no_grad():
            clean_emb = model.embed(clean_graph, feat).cpu().numpy()
        clean_emb = normalize(clean_emb, norm="l2")

        X_train, y_train = clean_emb[train_np], y_int[train_np]
        X_val, y_val = clean_emb[val_np], y_int[val_np]
        X_test_clean, y_test = clean_emb[test_np], y_int[test_np]

        clf = fit_lr_best_c(X_train, y_train, X_val, y_val)

        clean_acc = clf.score(X_test_clean, y_test) * 100.0
        all_results.append({
            "split_idx": split_idx, "sev": 0, "p": 0.0, "test_acc": clean_acc,
        })
        print(
            f"[ED_RAW] method=GraphMAE dataset={args.dataset} "
            f"split_idx={split_idx} seed={split_idx} sev=0 p=0.0 "
            f"test_acc={clean_acc:.4f}"
        )

        # Per-severity corrupted reps, frozen clf
        for sev_idx, p in SEVERITIES:
            dropped_ei = apply_edge_drop(base_ei_dev, num_nodes, p)
            noisy_graph = build_dgl_graph(dropped_ei, num_nodes, feat, y_device)
            with torch.no_grad():
                noisy_emb = model.embed(noisy_graph, feat).cpu().numpy()
            noisy_emb = normalize(noisy_emb, norm="l2")
            X_test_noisy = noisy_emb[test_np]
            noise_acc = clf.score(X_test_noisy, y_test) * 100.0
            all_results.append({
                "split_idx": split_idx, "sev": sev_idx, "p": p, "test_acc": noise_acc,
            })
            print(
                f"[ED_RAW] method=GraphMAE dataset={args.dataset} "
                f"split_idx={split_idx} seed={split_idx} sev={sev_idx} p={p} "
                f"test_acc={noise_acc:.4f}"
            )

    # ---------------- Aggregation + output ----------------
    print("\n=== GraphMAE Edge Deletion Results (aggregated over splits) ===")
    grouped = collections.defaultdict(list)
    for row in all_results:
        grouped[row["sev"]].append(row["test_acc"])

    label_for_sev = {0: "clean    "}
    for sev_idx, p in SEVERITIES:
        label_for_sev[sev_idx] = f"sev{sev_idx} p={p}"

    aggregated = {}
    for sev in sorted(grouped.keys()):
        accs = np.array(grouped[sev], dtype=np.float64)
        mean, std = accs.mean(), accs.std()
        aggregated[sev] = f"{mean:.2f} ± {std:.2f}"
        print(f"  {label_for_sev[sev]:<14}  {aggregated[sev]}")

    print(
        f"[ED_AGG] method=GraphMAE dataset={args.dataset} "
        f"clean=\"{aggregated.get(0, '')}\" "
        f"sev1=\"{aggregated.get(1, '')}\" "
        f"sev2=\"{aggregated.get(2, '')}\" "
        f"sev3=\"{aggregated.get(3, '')}\" "
        f"sev4=\"{aggregated.get(4, '')}\" "
        f"sev5=\"{aggregated.get(5, '')}\""
    )


if __name__ == "__main__":
    main()
