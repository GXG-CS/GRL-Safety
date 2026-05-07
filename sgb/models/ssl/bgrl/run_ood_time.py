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
from sgb.data.ood_splits import build_time_shift_split


NODE_DATASETS = {"arxiv"}  # time shift only available on arxiv

SPLIT_SEEDS_DEFAULT = [0]
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
flags.DEFINE_integer("train_max_year", 0, "Year cutoff train<=N")
flags.DEFINE_integer("ood_min_year", 0, "Year cutoff ood>=N")


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

    # Time-shift: require node_year
    if not hasattr(data, "node_year") or data.node_year is None:
        raise RuntimeError(f"Dataset has no node_year; time shift N/A.")
    year_t = data.node_year
    
    for split_seed in split_seeds:
        five_way = build_time_shift_split(
            dataset_name=FLAGS.dataset,
            year_tensor=year_t,
            labels=y,
            split_seed=split_seed,
            train_max_year=FLAGS.train_max_year if FLAGS.train_max_year > 0 else None,
            ood_min_year=FLAGS.ood_min_year if FLAGS.ood_min_year > 0 else None
        )
        meta = five_way["meta"]
        if meta.get("time_shift") == "not_applicable":
            print(f"[OOD_SKIP] method=BGRL_FT dataset={FLAGS.dataset} "
                  f"split_seed={split_seed} reason={meta.get('reason', 'unknown')} "
                  f"num_classes={meta.get('num_classes')}")
            continue

        print(
            f"[OOD_SPLIT] dataset={FLAGS.dataset} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_5_35_ascending_year')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_year_range={meta['train_pool_year_range']} "
            f"ood_val_year_range={meta['ood_val_year_range']} "
            f"ood_test_year_range={meta['ood_test_year_range']} "
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
                f"shift=time selector=id_val "
                f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}"
            )

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(
                f"[OOD_ORACLE] method=BGRL_FT dataset={FLAGS.dataset} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=time selector=ood_val "
                f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}"
            )


if __name__ == "__main__":
    app.run(main)
