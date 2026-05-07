"""Linear-probe eval of a GraphMAE joint-pretrained checkpoint.

Loads a PreModel with the exact same architecture used by
`pretrain_joint.py` (GAT encoder + MLP decoder, no layernorm, drop_edge=0),
loads the joint-pretrained state dict, extracts encoder embeddings for the
target dataset, and fits a logistic regression head per split.

Usage:
    python sgb/models/ssl/graphmae/eval_joint.py \
        --dataset cora \
        --ckpt_path ckpts/graphmae/all/model.pt

Prints `[JOINT_FN]` style raw lines and a `[JOINT_AGG]` aggregated line.
"""

import os
import os.path as osp
import sys
import argparse
import collections

import numpy as np
import torch
import dgl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sgb.data.tag_registry import load as load_tag  # type: ignore
from graphmae.models.edcoder import PreModel  # type: ignore


# -----------------------------------------------------------------------------
# Model builder — must mirror pretrain_joint.py's PreModel args exactly
# -----------------------------------------------------------------------------
def build_joint_model(
    num_features=768,
    num_hidden=768,
    num_layers=2,
    num_heads=4,
    mask_rate=0.5,
    replace_rate=0.0,
    loss_fn="sce",
    alpha_l=3.0,
):
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
# Data: load a single target dataset as DGL (no multi-source batching)
# -----------------------------------------------------------------------------
def _materialize_feat(data):
    """Same logic as pretrain_joint._materialize_node_feat."""
    if data.x is not None and data.x.dtype == torch.long and data.x.ndim == 1:
        return data.node_text_feat[data.x].float()
    return data.node_text_feat.float()


def _extract_splits(data, n_max=5):
    """Pull out up to 5 (train, val, test) mask tuples from whatever split
    representation the dataset has."""
    splits = []
    if hasattr(data, "train_masks") and data.train_masks is not None:
        n = min(n_max, len(data.train_masks))
        for i in range(n):
            splits.append({
                "train": data.train_masks[i],
                "val": data.val_masks[i],
                "test": data.test_masks[i],
            })
    elif hasattr(data, "train_mask") and data.train_mask is not None:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            n = min(n_max, tm.size(1))
            for i in range(n):
                splits.append({
                    "train": tm[:, i],
                    "val": vm[:, i],
                    "test": tsm[:, i] if tsm.dim() == 2 else tsm,
                })
        else:
            # Single fixed split; replicate 5x so aggregation still yields
            # mean/std (std will be 0 for clean LR since it's deterministic).
            for _ in range(n_max):
                splits.append({"train": tm, "val": vm, "test": tsm})
    return splits


def load_dgl_target(name):
    data, _ = load_tag(name)
    feat = _materialize_feat(data)

    src = data.edge_index[0].long()
    dst = data.edge_index[1].long()
    g = dgl.graph((src, dst), num_nodes=feat.size(0))
    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = feat

    y = data.y.squeeze() if data.y.dim() > 1 else data.y
    g.ndata["label"] = y

    splits = _extract_splits(data)
    return g, splits


# -----------------------------------------------------------------------------
# Linear probe helpers
# -----------------------------------------------------------------------------
def fit_lr_best_c(X_train, y_train, X_val, y_val):
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[JOINT] Using {device}")

    graph, splits = load_dgl_target(args.dataset)
    num_classes = int(graph.ndata["label"].max().item()) + 1
    print(
        f"[JOINT] Dataset: {args.dataset}, nodes={graph.num_nodes()}, "
        f"edges={graph.num_edges()}, classes={num_classes}, splits={len(splits)}"
    )

    graph = graph.to(device)
    feat = graph.ndata["feat"].to(device)
    y_int = graph.ndata["label"].cpu().numpy().astype(np.int64)

    model = build_joint_model(num_features=feat.size(1))
    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"[JOINT] Loaded ckpt from {args.ckpt_path}")

    # Encoder embeddings (full graph — cora is small)
    with torch.no_grad():
        emb = model.embed(graph, feat).cpu().numpy()
    emb = normalize(emb, norm="l2")

    test_accs = []
    for split_idx, split in enumerate(splits):
        train_np = split["train"].cpu().numpy().astype(bool)
        val_np = split["val"].cpu().numpy().astype(bool)
        test_np = split["test"].cpu().numpy().astype(bool)

        X_train, y_train = emb[train_np], y_int[train_np]
        X_val, y_val = emb[val_np], y_int[val_np]
        X_test, y_test = emb[test_np], y_int[test_np]

        clf = fit_lr_best_c(X_train, y_train, X_val, y_val)
        acc = clf.score(X_test, y_test) * 100.0
        test_accs.append(acc)
        print(
            f"[JOINT_RAW] method=GraphMAE dataset={args.dataset} "
            f"split_idx={split_idx} test_acc={acc:.4f}"
        )

    arr = np.array(test_accs, dtype=np.float64)
    mean, std = arr.mean(), arr.std()
    print(
        f"[JOINT_AGG] method=GraphMAE dataset={args.dataset} "
        f"test=\"{mean:.2f} ± {std:.2f}\""
    )


if __name__ == "__main__":
    main()
