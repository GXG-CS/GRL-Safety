"""GNN baseline (GCN/GAT/SAGE) from-scratch FT + node-level OOD eval.

Mirrors BGRL's run_ood_degree.py protocol. Degree-based covariate shift
following GOOD's 10/1/1 protocol. Uses random-init encoder (no pretrain).

Tracks TWO best checkpoints:
  - best_id_val → main protocol ([OOD_RAW])
  - best_ood_val → appendix oracle ([OOD_ORACLE])
"""

import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from sklearn.metrics import f1_score

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.supervised.gnn_baseline import GNNEncoderWrapper, METHOD_NAMES
from sgb.data.tag_registry import load as load_tag


NODE_DATASETS = {
    "cora", "citeseer", "pubmed", "wikics", "arxiv", "arxiv23",
    "elephoto", "elecomp", "tolokers", "dblp", "amazonratings",
    "bookhis", "bookchild", "sportsfit", "products",
}

SPLIT_SEEDS_DEFAULT = [0, 1, 2, 3, 4]
RUN_SEEDS_DEFAULT = [42, 43, 44, 45, 46]


# -----------------------------------------------------------------------------
# Degree split builder (GOOD 60/20/20 descending, inlined per-method)
# -----------------------------------------------------------------------------

def _compute_node_degree(edge_index, num_nodes):
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg


def build_degree_split(dataset_name, edge_index, labels, split_seed):
    """Return a 5-way degree-OOD split matching GOOD's covariate shift."""
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


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "TAG node dataset name.")
flags.DEFINE_string("model", "gcn", "GNN baseline: gcn, gat, or sage.")
flags.DEFINE_integer("hidden", 768, "Hidden dim.")
flags.DEFINE_integer("num_layers", 2, "Number of encoder layers.")
flags.DEFINE_integer("max_epochs", 500, "Max FT epochs per run.")
flags.DEFINE_integer("patience", 200, "Early-stop patience (id_val).")
flags.DEFINE_float("lr", 1e-3, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-4, "Weight decay.")
flags.DEFINE_float("dropout", 0.2, "Dropout.")
flags.DEFINE_bool("debug", False,
                  "If True, collapse to split_seeds=[0] run_seeds=[42].")


class FTModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        h = self.encoder(data)
        h = self.dropout(h)
        return self.head(h)


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


def _build_encoder(in_channels):
    return GNNEncoderWrapper(
        model_name=FLAGS.model,
        in_channels=in_channels,
        hidden_channels=FLAGS.hidden,
        num_layers=FLAGS.num_layers,
        dropout=FLAGS.dropout,
    )


def method_tag():
    return METHOD_NAMES[FLAGS.model.lower()]


def _train_one_run(data, y, masks, num_classes, input_size, run_seed, device):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    encoder = _build_encoder(input_size)
    model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    best_id_val = -float("inf")
    best_ood_val = -float("inf")
    best_id_state = None
    best_ood_state = None
    no_improve_id = 0

    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
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

        if no_improve_id >= FLAGS.patience:
            print(f"Early stop at epoch {epoch} (no id_val improvement for {FLAGS.patience})")
            break

    def _eval_with_state(state):
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = model(data)
            return {
                "id_val":     _accuracy(logits, y, masks["id_val"]),
                "id_test":    _accuracy(logits, y, masks["id_test"]),
                "id_test_f1": _macro_f1(logits, y, masks["id_test"]),
                "ood_val":    _accuracy(logits, y, masks["ood_val"]),
                "ood_test":   _accuracy(logits, y, masks["ood_test"]),
                "ood_test_f1": _macro_f1(logits, y, masks["ood_test"]),
            }

    main_metrics = _eval_with_state(best_id_state) if best_id_state is not None \
        else {"id_test": float("nan"), "ood_test": float("nan"),
              "id_val": best_id_val, "ood_val": 0.0,
              "id_test_f1": 0.0, "ood_test_f1": 0.0}
    oracle_metrics = _eval_with_state(best_ood_state) if best_ood_state is not None \
        else main_metrics

    return main_metrics, oracle_metrics


def _gap(id_v, ood_v):
    if id_v is None or ood_v is None or id_v != id_v or ood_v != ood_v:
        return (float("nan"),) * 3
    gap_abs = id_v - ood_v
    gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
    rr = ood_v / id_v if id_v > 0 else 0.0
    return gap_abs, gap_rel, rr


def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tag = method_tag()
    print(f"[{tag} FT-OOD] Using {device}")

    if FLAGS.dataset not in NODE_DATASETS:
        raise ValueError(
            f"run_ood_degree.py currently supports node datasets only; "
            f"got {FLAGS.dataset}. Supported: {sorted(NODE_DATASETS)}"
        )

    data, _ = load_tag(FLAGS.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, "node_text_feat"):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, "node_text_feat"):
        data.x = data.node_text_feat
    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    input_size = data.x.size(1)
    print(f"[{tag} FT-OOD] {FLAGS.dataset}, N={data.num_nodes}, "
          f"E={data.edge_index.size(1)}, C={num_classes}")

    split_seeds = list(SPLIT_SEEDS_DEFAULT)
    run_seeds = list(RUN_SEEDS_DEFAULT)
    if FLAGS.debug:
        split_seeds = split_seeds[:1]
        run_seeds = run_seeds[:1]
        print(f"[OOD_SMOKE] debug mode: split_seeds={split_seeds} run_seeds={run_seeds}")

    for split_seed in split_seeds:
        five_way = build_degree_split(
            dataset_name=FLAGS.dataset,
            edge_index=data.edge_index,
            labels=y,
            split_seed=split_seed,
        )
        meta = five_way["meta"]
        if meta.get("degree_shift") == "not_applicable":
            print(f"[OOD_SKIP] method={tag} dataset={FLAGS.dataset} "
                  f"split_seed={split_seed} reason={meta.get('reason', 'unknown')} "
                  f"num_classes={meta.get('num_classes')}")
            continue

        print(
            f"[OOD_SPLIT] dataset={FLAGS.dataset} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_20_20_descending')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_deg_range={meta['train_pool_degree_range']} "
            f"ood_val_deg_range={meta['ood_val_degree_range']} "
            f"ood_test_deg_range={meta['ood_test_degree_range']} "
            f"smallest_train_pool_class={meta['smallest_train_pool_class']}"
        )

        N = data.num_nodes
        masks = {
            "train":    _idx_to_mask(five_way["train"],    N, device),
            "id_val":   _idx_to_mask(five_way["id_val"],   N, device),
            "id_test":  _idx_to_mask(five_way["id_test"],  N, device),
            "ood_val":  _idx_to_mask(five_way["ood_val"],  N, device),
            "ood_test": _idx_to_mask(five_way["ood_test"], N, device),
        }

        for run_seed in run_seeds:
            main_m, oracle_m = _train_one_run(
                data=data, y=y, masks=masks,
                num_classes=num_classes, input_size=input_size,
                run_seed=run_seed, device=device,
            )

            gA, gR, rR = _gap(main_m["id_test"], main_m["ood_test"])
            print(
                f"[OOD_RAW] method={tag} dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=id_val "
                f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}"
            )

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(
                f"[OOD_ORACLE] method={tag} dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=ood_val "
                f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}"
            )


if __name__ == "__main__":
    app.run(main)
