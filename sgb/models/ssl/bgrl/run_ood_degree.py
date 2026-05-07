"""BGRL FT + node-level OOD evaluation (GFM-Safety Dim 2, node only).

Degree-based covariate shift following GOOD's 10/1/1 protocol. See
`experiment_design/ood/ood_experiment_design.md` for the full spec.

Protocol:
  1. Load the frozen degree split artifact from
     `experiment_design/ood/splits/node_degree/<dataset>_seed<k>.pt`
     (or build it on first use via sgb.data.ood_splits).
  2. Initialize FTModel = BGRL encoder + linear head.
  3. Fine-tune on `train` only; per epoch, eval id_val / id_test / ood_val / ood_test.
  4. Track TWO best checkpoints simultaneously:
       - best_id_val → main protocol (`[OOD_RAW]`, deployment-realistic)
       - best_ood_val → appendix oracle (`[OOD_ORACLE]`, GOOD-comparable)
  5. At the end of each run, reload each best checkpoint and log both streams.
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

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN
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
#
# Self-contained copy of the degree-OOD split. Matches the reference
# implementation in GOOD.data.good_datasets.good_cora.get_covariate_shift_graph:
# sort descending by degree, 60% / 20% / 20% train / ood_val / ood_test, then
# random-shuffle (seeded) the train slice to carve id_val / id_test (each 10%
# of total). No disk caching — deterministic and cheap to rebuild per call.

def _compute_node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Undirected-style node degree (each edge contributes to both endpoints)."""
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg


def build_degree_split(
    dataset_name: str,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    split_seed: int,
):
    """Return a 5-way degree-OOD split matching GOOD's covariate shift.

    Output keys: `train`, `id_val`, `id_test`, `ood_val`, `ood_test` (1-D
    `torch.long` tensors of global node indices) and a `meta` sub-dict.
    Returns a stub with `meta.degree_shift = "not_applicable"` only if any top
    bucket would be empty.
    """
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


FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "TAG node dataset name.")
flags.DEFINE_string("ckpt_path", None, "Pretrained BGRL encoder .pt.")
flags.DEFINE_multi_integer("graph_encoder_layer", [768, 768], "Encoder layers.")
flags.DEFINE_integer("max_epochs", 1000, "Max FT epochs per run.")
flags.DEFINE_integer("patience", 200, "Early-stop patience (id_val).")
flags.DEFINE_float("lr", 5e-4, "Learning rate.")
flags.DEFINE_float("weight_decay", 1e-5, "Weight decay.")
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


def _idx_to_mask(idx: torch.Tensor, N: int, device) -> torch.Tensor:
    m = torch.zeros(N, dtype=torch.bool, device=device)
    if idx.numel() > 0:
        m[idx.to(device)] = True
    return m


def _accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    pred = logits[mask].argmax(-1)
    return (pred == y[mask]).float().mean().item() * 100.0


def _macro_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return float("nan")
    pred = logits[mask].argmax(-1).cpu().numpy()
    true = y[mask].cpu().numpy()
    return f1_score(true, pred, average="macro") * 100.0


def _train_one_run(
    data,
    y,
    masks,                 # dict with train/id_val/id_test/ood_val/ood_test bool masks
    num_classes: int,
    input_size: int,
    ckpt_state,
    run_seed: int,
    device,
):
    """One FT run on the current degree split. Returns a dict with main and
    oracle metrics. Tracks TWO best checkpoints (by id_val and ood_val)."""
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    encoder = GCN([input_size] + list(FLAGS.graph_encoder_layer), batchnorm=True)
    encoder.load_state_dict(ckpt_state)
    model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay,
    )

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

        # Track best-by-id_val (main protocol; early stops by this)
        if id_val_acc > best_id_val:
            best_id_val = id_val_acc
            best_id_state = copy.deepcopy(model.state_dict())
            no_improve_id = 0
        else:
            no_improve_id += 1

        # Track best-by-ood_val (oracle; does NOT drive early stopping)
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
                "id_val":    _accuracy(logits, y, masks["id_val"]),
                "id_test":   _accuracy(logits, y, masks["id_test"]),
                "id_test_f1": _macro_f1(logits, y, masks["id_test"]),
                "ood_val":   _accuracy(logits, y, masks["ood_val"]),
                "ood_test":  _accuracy(logits, y, masks["ood_test"]),
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
    if id_v is None or ood_v is None or id_v != id_v or ood_v != ood_v:  # NaN check
        return (float("nan"),) * 3
    gap_abs = id_v - ood_v
    gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
    rr = ood_v / id_v if id_v > 0 else 0.0
    return gap_abs, gap_rel, rr


def main(argv):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[BGRL FT-OOD] Using {device}")

    if FLAGS.dataset not in NODE_DATASETS:
        raise ValueError(
            f"BGRL run_ood_degree.py currently supports node datasets only; "
            f"got {FLAGS.dataset}. Supported: {sorted(NODE_DATASETS)}"
        )

    # -------- Load data via tag_registry (unified interface) --------
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
    print(f"[BGRL FT-OOD] {FLAGS.dataset}, N={data.num_nodes}, "
          f"E={data.edge_index.size(1)}, C={num_classes}")

    ckpt_state = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and "model" in ckpt_state:
        ckpt_state = ckpt_state["model"]

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
            print(f"[OOD_SKIP] method=BGRL_FT dataset={FLAGS.dataset} "
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

        # Convert idx tensors to bool masks on the device.
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
                ckpt_state=ckpt_state, run_seed=run_seed, device=device,
            )

            gA, gR, rR = _gap(main_m["id_test"], main_m["ood_test"])
            print(
                f"[OOD_RAW] method=BGRL_FT dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=id_val "
                f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}"
            )

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(
                f"[OOD_ORACLE] method=BGRL_FT dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=degree selector=ood_val "
                f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}"
            )


if __name__ == "__main__":
    app.run(main)
