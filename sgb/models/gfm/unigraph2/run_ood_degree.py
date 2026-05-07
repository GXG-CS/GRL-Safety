"""UniGraph2 FT + node-level OOD evaluation (GFM-Safety Dim 2, node only).

Parallel to sgb/models/ssl/graphmae/run_ood_degree.py. Degree-based covariate shift
following GOOD's 60/20/20 descending protocol with inlined `build_degree_split`.
Loads a joint-pretrained UniGraph2 encoder, attaches a linear head, fine-tunes
end-to-end with id_val-based model selection, also tracks best-by-ood_val as
oracle stream. Emits `[OOD_RAW]` (main) and `[OOD_ORACLE]` (appendix) lines.
"""

import copy
import os
import os.path as osp
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score

_UG2_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_UG2_DIR, "..", "..", ".."))
if _UG2_DIR not in sys.path:
    sys.path.insert(0, _UG2_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.unigraph2 import UniGraph2  # type: ignore
from sgb.data.tag_registry import load as load_tag


NODE_DATASETS = {
    "cora", "citeseer", "pubmed", "wikics", "arxiv", "arxiv23",
    "elephoto", "elecomp", "tolokers", "dblp", "amazonratings",
    "bookhis", "bookchild", "sportsfit", "products",
}

SPLIT_SEEDS_DEFAULT = [0, 1, 2, 3, 4]
RUN_SEEDS_DEFAULT = [42, 43, 44, 45, 46]


# -----------------------------------------------------------------------------
# Degree split builder (self-contained copy of GraphMAE's version)
# -----------------------------------------------------------------------------
def _compute_node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg


def build_degree_split(dataset_name, edge_index, labels, split_seed):
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

    if (
        train_pool_idx.numel() == 0
        or ood_val_idx.numel() == 0
        or ood_test_idx.numel() == 0
    ):
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
        id_val_idx = shuffled[-2 * num_id : -num_id]
        id_test_idx = shuffled[-num_id :]

    train_pool_counts = torch.bincount(train_pool_y, minlength=num_classes_total)
    present = int((train_pool_counts > 0).sum().item())
    smallest = int(train_pool_counts[train_pool_counts > 0].min().item()) \
        if present > 0 else 0
    missing_classes = int(num_classes_total - present)

    def _range_tuple(pool_deg):
        if pool_deg.numel() == 0:
            return (None, None)
        return (int(pool_deg.min().item()), int(pool_deg.max().item()))

    if missing_classes > 0:
        print(
            f"[OOD_WARN] dataset={dataset_name} split_seed={split_seed} "
            f"missing_classes_in_train_pool={missing_classes} "
            f"(out of {num_classes_total}); proceeding (matches GOOD)."
        )

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


def build_model(num_features=768, num_hidden=768, num_layers=3,
                num_experts=8, num_selected_experts=2):
    return UniGraph2(
        input_dims={"text": num_features},
        hidden_dim=num_hidden,
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        num_layers=num_layers,
        feat_drop_rate=0.1,
        edge_mask_rate=0.1,
        gamma=2.0,
        lambda_spd=0.5,
    )


class FTModel(nn.Module):
    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)

    def forward(self, g, x):
        h = self.pre_model(g, {"text": x}, spd_matrix=None, return_embeddings=True)
        h = self.dropout(h)
        return self.head(h)


def _load_data(name, device):
    data, _ = load_tag(name)

    if data.x is not None and data.x.dtype == torch.long and hasattr(data, "node_text_feat"):
        feat = data.node_text_feat[data.x].float()
    elif data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        feat = data.x.float()
    elif hasattr(data, "node_text_feat"):
        feat = data.node_text_feat.float()
    else:
        raise RuntimeError(f"Cannot extract 768d features for {name}")

    y = data.y.squeeze() if data.y is not None and data.y.dim() > 1 else data.y

    raw_ei = data.edge_index.long()
    src, dst = raw_ei[0], raw_ei[1]
    non_self = src != dst
    base_ei = raw_ei[:, non_self]

    return feat.to(device), y.long().to(device), base_ei.to(device)


def _make_dgl_graph(edge_index, num_nodes, device):
    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    g = g.remove_self_loop().add_self_loop()
    return g.to(device)


def _idx_to_mask(idx, N, device):
    m = torch.zeros(N, dtype=torch.bool, device=device)
    if idx.numel() > 0:
        m[idx.to(device)] = True
    return m


def _accuracy(logits, y, mask):
    if mask.sum() == 0:
        return float("nan")
    pred = logits[mask].argmax(-1)
    return (pred == y[mask]).float().mean().item() * 100.0


def _macro_f1(logits, y, mask):
    if mask.sum() == 0:
        return float("nan")
    pred = logits[mask].argmax(-1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return f1_score(true, pred, average="macro") * 100.0


def _train_one_run(g, feat, y, masks, num_classes, state, args, run_seed, device):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    pre_model = build_model(num_features=feat.size(1))
    pre_model.load_state_dict(state, strict=False)
    model = FTModel(
        pre_model, num_hidden=768, num_classes=num_classes, dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    best_id_val = -float("inf")
    best_ood_val = -float("inf")
    best_id_state = None
    best_ood_state = None
    no_improve_id = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(g, feat)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(g, feat)
            id_val_acc = _accuracy(logits, y, masks["id_val"])
            ood_val_acc = _accuracy(logits, y, masks["ood_val"])

        if id_val_acc > best_id_val:
            best_id_val = id_val_acc
            best_id_state = copy.deepcopy(model.state_dict())
            no_improve_id = 0
        else:
            no_improve_id += 1

        if ood_val_acc > best_ood_val:
            best_ood_val = ood_val_acc
            best_ood_state = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % 50 == 0:
            id_test_acc = _accuracy(logits, y, masks["id_test"])
            ood_test_acc = _accuracy(logits, y, masks["ood_test"])
            print(
                f"[epoch {epoch:4d}] loss={loss.item():.4f} "
                f"id_val={id_val_acc:.2f} ood_val={ood_val_acc:.2f} "
                f"id_test={id_test_acc:.2f} ood_test={ood_test_acc:.2f}"
            )

        if no_improve_id >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break

    def _eval_with_state(state_dict):
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            logits = model(g, feat)
            return {
                "id_val":    _accuracy(logits, y, masks["id_val"]),
                "id_test":   _accuracy(logits, y, masks["id_test"]),
                "id_test_f1": _macro_f1(logits, y, masks["id_test"]),
                "ood_val":   _accuracy(logits, y, masks["ood_val"]),
                "ood_test":  _accuracy(logits, y, masks["ood_test"]),
                "ood_test_f1": _macro_f1(logits, y, masks["ood_test"]),
            }

    main_m = _eval_with_state(best_id_state) if best_id_state is not None \
        else {"id_test": float("nan"), "ood_test": float("nan"),
              "id_val": best_id_val, "ood_val": 0.0,
              "id_test_f1": 0.0, "ood_test_f1": 0.0}
    oracle_m = _eval_with_state(best_ood_state) if best_ood_state is not None else main_m
    return main_m, oracle_m


def _gap(id_v, ood_v):
    if id_v is None or ood_v is None or id_v != id_v or ood_v != ood_v:
        return (float("nan"),) * 3
    gap_abs = id_v - ood_v
    gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
    rr = ood_v / id_v if id_v > 0 else 0.0
    return gap_abs, gap_rel, rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--patience", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.dataset not in NODE_DATASETS:
        raise ValueError(
            f"UniGraph2 run_ood_degree.py currently supports node datasets only; "
            f"got {args.dataset}. Supported: {sorted(NODE_DATASETS)}"
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[UG2 FT-OOD] Using {device}")

    feat, y, base_ei = _load_data(args.dataset, device)
    num_nodes = int(feat.size(0))
    num_classes = int(y.max().item()) + 1
    g = _make_dgl_graph(base_ei, num_nodes, device)
    print(
        f"[UG2 FT-OOD] {args.dataset}, N={num_nodes}, "
        f"E={base_ei.size(1)}, C={num_classes}"
    )

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    split_seeds = list(SPLIT_SEEDS_DEFAULT)
    run_seeds = list(RUN_SEEDS_DEFAULT)
    if args.debug:
        split_seeds = split_seeds[:1]
        run_seeds = run_seeds[:1]
        print(f"[OOD_SMOKE] debug mode: split_seeds={split_seeds} run_seeds={run_seeds}")

    for split_seed in split_seeds:
        five_way = build_degree_split(
            dataset_name=args.dataset,
            edge_index=base_ei,
            labels=y,
            split_seed=split_seed,
        )
        meta = five_way["meta"]
        if meta.get("degree_shift") == "not_applicable":
            print(
                f"[OOD_SKIP] method=UniGraph2_FT dataset={args.dataset} "
                f"split_seed={split_seed} reason={meta.get('reason', 'unknown')} "
                f"num_classes={meta.get('num_classes')}"
            )
            continue

        print(
            f"[OOD_SPLIT] dataset={args.dataset} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_20_20_descending')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_deg_range={meta['train_pool_degree_range']} "
            f"ood_val_deg_range={meta['ood_val_degree_range']} "
            f"ood_test_deg_range={meta['ood_test_degree_range']} "
            f"smallest_train_pool_class={meta['smallest_train_pool_class']}"
        )

        masks = {
            "train":    _idx_to_mask(five_way["train"],    num_nodes, device),
            "id_val":   _idx_to_mask(five_way["id_val"],   num_nodes, device),
            "id_test":  _idx_to_mask(five_way["id_test"],  num_nodes, device),
            "ood_val":  _idx_to_mask(five_way["ood_val"],  num_nodes, device),
            "ood_test": _idx_to_mask(five_way["ood_test"], num_nodes, device),
        }

        for run_seed in run_seeds:
            main_m, oracle_m = _train_one_run(
                g=g, feat=feat, y=y, masks=masks,
                num_classes=num_classes, state=state, args=args,
                run_seed=run_seed, device=device,
            )

            gA, gR, rR = _gap(main_m["id_test"], main_m["ood_test"])
            print(
                f"[OOD_RAW] method=UniGraph2_FT dataset={args.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=id_val "
                f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}"
            )

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(
                f"[OOD_ORACLE] method=UniGraph2_FT dataset={args.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=ood_val "
                f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}"
            )


if __name__ == "__main__":
    main()
