#!/usr/bin/env python
# coding: utf-8
"""
GFT fine-tuning + within-dataset OOD evaluation (GFM-Safety Dim 2).

Node shift: degree-based 5-way split (train / id_val / id_test / ood_val / ood_test)
following `experiment_design/ood/ood_experiment_design.md`:

  - ID pool:  Q20 <= deg(v) <= Q80   (60/20/20 stratified by class)
  - OOD pool: deg(v) < Q20 or > Q80  (50/50 random)
  - Feasibility fallback: Q20/Q80 -> Q15/Q85 -> Q10/Q90 -> not_applicable

Training protocol:
  - Main (deployment-realistic): select best checkpoint by `id_val`
    -> reported via [OOD_RAW]  (main-table stream)
  - Appendix (oracle ceiling):    track best checkpoint by `ood_val`
    -> reported via [OOD_ORACLE] (appendix-only stream)

Evaluation: once per (split_seed, run_seed), both checkpoints are loaded
back and evaluated on `id_test` and `ood_test` under the shared encoder.

Node-only for this pilot; molecule scaffold OOD is a separate follow-up file.
"""

import os
import os.path as osp
import sys
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW

from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.args import get_args_finetune
from utils.others import seed_everything, load_params

from task.node import ft_node, eval_node

import warnings
import wandb

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Scope (node only)
# -----------------------------------------------------------------------------

NODE_DATASETS = {
    "cora", "citeseer", "pubmed", "wikics", "arxiv", "arxiv23",
    "elephoto", "elecomp", "tolokers", "dblp", "amazonratings",
    "bookhis", "bookchild", "sportsfit", "products",
}

SPLIT_SEEDS_DEFAULT = [0, 1, 2, 3, 4]
RUN_SEEDS_DEFAULT = [42, 43, 44, 45, 46]

# -----------------------------------------------------------------------------
# Degree split builder (GOOD 60/20/20 descending — exact port of
# GOOD.data.good_datasets.good_cora.get_covariate_shift_graph)
# -----------------------------------------------------------------------------
#
# This is an inlined, self-contained implementation so the GFT method file is
# not coupled to a shared module. BGRL / GraphMAE / GIT keep their own copy of
# the same function. Split is computed fresh every call (no disk cache) — the
# algorithm is deterministic per split_seed and cheap enough (< 3 s on arxiv).
#
# Protocol (matches GOOD paper exactly):
#   1. Sort labeled nodes by degree DESCENDING (GOOD: `sorted_data_list[::-1]`).
#   2. First slice: 60% train / 20% ood_val / 20% ood_test by position.
#      -> train = HIGHEST-degree 60%
#      -> ood_val = middle 20% by degree
#      -> ood_test = LOWEST-degree 20%   (cold-start test)
#   3. From the train slice, random-shuffle (seeded per split_seed) and carve:
#      -> id_val  = 10 % of total  (from train pool, sampled)
#      -> id_test = 10 % of total  (from train pool, sampled)
#      -> actual train = the remaining 40 % of total
#   Final 5-way allocation on N labeled nodes:
#      40 % train  / 10 % id_val / 10 % id_test / 20 % ood_val / 20 % ood_test

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
    `torch.long` tensors of *global* node indices), plus a `meta` sub-dict.
    Returns a stub with `meta.degree_shift = "not_applicable"` only if one of
    the three top-level buckets would be empty.
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

    # Sort labeled nodes by degree DESCENDING, with node id as deterministic
    # tiebreak so two machines always produce the same ordering.
    # `descending=True` corresponds to GOOD's `sorted_data_list[::-1]`.
    sort_key = labeled_deg * (num_nodes + 1) + labeled_idx_all.long()
    order = torch.argsort(sort_key, descending=True)
    sorted_idx = labeled_idx_all[order]
    sorted_y = labeled_y_all[order]
    sorted_deg = labeled_deg[order]

    n_labeled = int(sorted_idx.numel())

    # First slice: 60% train / 20% ood_val / 20% ood_test
    train_end = int(round(n_labeled * 0.60))
    ood_val_end = int(round(n_labeled * 0.80))

    train_pool_idx = sorted_idx[:train_end]     # HIGHEST 60% by degree
    train_pool_y = sorted_y[:train_end]
    ood_val_idx = sorted_idx[train_end:ood_val_end]   # middle 20%
    ood_test_idx = sorted_idx[ood_val_end:]            # LOWEST 20% (cold start)

    # Early-exit: any empty bucket makes the split unusable.
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
                "dataset": dataset_name,
                "split_seed": split_seed,
                "shift": "degree",
                "strategy": "good_60_20_20_descending",
                "degree_shift": "not_applicable",
                "reason": "empty_bucket",
                "num_classes": int(num_classes_total),
                "num_nodes_total": int(num_nodes),
                "n_labeled": int(n_labeled),
            },
        }

    # Second slice: from train_pool, shuffle (seeded) and carve
    #   id_val  = last 2*num_id..last num_id
    #   id_test = last num_id
    #   actual train = everything before these
    # Matches GOOD: `train_list[:-2*num_id_test], [-2:-1], [-1:]` after
    # `random.shuffle(train_list)`.
    num_id = int(round(n_labeled * 0.10))   # 10 % of total labeled nodes
    if 2 * num_id >= train_pool_idx.numel():
        # Train pool too small to carve id splits — degenerate case; fall back
        # to using the whole train_pool as train with empty id_val/id_test.
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

    # Metadata for logging / audit
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


# -----------------------------------------------------------------------------
# Eval helpers
# -----------------------------------------------------------------------------

def _train_split_view(five_way):
    """The 3-key dict passed into ft_node / eval_node during training.

    `valid` = id_val so that early stopping and per-epoch logging follow the
    main (id_val-selected) protocol. `test` = id_test but the loss never
    touches test, so this is purely for bookkeeping.
    """
    return {
        "train": five_way["train"],
        "valid": five_way["id_val"],
        "test": five_way["id_test"],
    }


def _ood_view(five_way):
    """A 3-key dict whose `test` is the OOD-test pool, so a single eval_node
    call returns accuracy on the OOD distribution. `train` is reused as the
    ID-train pool (needed for prototype computation), `valid` as ood_val
    (so we can simultaneously track the oracle selector)."""
    return {
        "train": five_way["train"],
        "valid": five_way["ood_val"],
        "test": five_way["ood_test"],
    }


def _eval_both(task_model, data_or_dataset, loader, five_way, labels, params,
               num_neighbors):
    """Run `eval_node` twice to get id_val / id_test / ood_val / ood_test.

    Full-batch encoding is cheap so doing this twice per epoch is fine on
    all current node datasets except (maybe) products. If products turns
    out to be the bottleneck we can refactor to a single encode + 4 masks.
    """
    id_view = _train_split_view(five_way)
    ood_view = _ood_view(five_way)

    # eval_node does not wrap itself in torch.no_grad; calling it twice per
    # epoch under the default autograd mode doubles activation memory and
    # pushes big graphs (arxiv, products, ...) OOM. We explicitly suppress
    # gradients and empty the cache between the two calls.
    with torch.no_grad():
        r_id = eval_node(
            model=task_model, dataset=data_or_dataset, loader=loader,
            split=id_view, labels=labels, params=params,
            num_neighbors=num_neighbors,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        r_ood = eval_node(
            model=task_model, dataset=data_or_dataset, loader=loader,
            split=ood_view, labels=labels, params=params,
            num_neighbors=num_neighbors,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "train": float(r_id["train"]),
        "id_val": float(r_id["val"]),
        "id_test": float(r_id["test"]),
        "id_test_f1": float(r_id.get("test_f1", 0.0)),
        "ood_val": float(r_ood["val"]),
        "ood_test": float(r_ood["test"]),
        "ood_test_f1": float(r_ood.get("test_f1", 0.0)),
    }


# -----------------------------------------------------------------------------
# Main run loop (adapted from run_edge_deletion.py but node-only)
# -----------------------------------------------------------------------------

def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")

    data_name = params["finetune_dataset"]
    if data_name not in NODE_DATASETS:
        raise ValueError(
            f"run_ood_degree.py currently supports node datasets only; "
            f"got {data_name}. Supported: {sorted(NODE_DATASETS)}"
        )
    params["task"] = "node"

    # ---- Load TAG data via tag_registry ----
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag
    data, _ = load_tag(data_name)

    if data.x is None or data.x.ndim == 2:
        data.x = torch.arange(data.node_text_feat.size(0), dtype=torch.long)
    if not hasattr(data, "xe") or data.get("xe") is None:
        data.xe = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    labels = data.y.squeeze()
    num_classes = int(labels.max().item()) + 1
    params["num_classes"] = num_classes
    data.y = labels

    # Inline preprocessing above already mirrors GFT's `span_node_and_edge_idx`
    # (x -> 1D arange, xe -> zero for single edge type), so we do NOT call
    # `pre_node` / `filter_unnecessary_attrs` here — they require an
    # InMemoryDataset with a `.data` attribute that we do not own.

    # ---- Build encoder + vq + load pretrained ckpt (same as edge_deletion) ----
    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )
    vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    ckpt_dir = params.get("ckpt_dir")
    if ckpt_dir is None and params.get("pretrain_dataset", "all") != "na":
        ckpt_dir = osp.abspath(osp.join(
            osp.dirname(__file__), "..", "..", "..", "..", "ckpts", "GFT",
        ))
    if ckpt_dir:
        encoder = load_params(encoder, osp.join(ckpt_dir, "encoder.pt"))
        vq = load_params(vq, osp.join(ckpt_dir, "vq.pt"))
        print(f"Loaded pretrained encoder and vq from {ckpt_dir}")

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    # ---- Decide seeds (smoke collapses to 1x1 via --debug) ----
    split_seeds = list(params.get("split_seeds", SPLIT_SEEDS_DEFAULT))
    run_seeds = list(params.get("seeds", RUN_SEEDS_DEFAULT))
    if params.get("debug", False):
        split_seeds = split_seeds[:1]
        run_seeds = run_seeds[:1]
        print(f"[OOD_SMOKE] debug mode: split_seeds={split_seeds} run_seeds={run_seeds}")

    all_rows = []

    for split_seed in split_seeds:
        five_way = build_degree_split(
            dataset_name=data_name,
            edge_index=data.edge_index,
            labels=labels,
            split_seed=split_seed,
        )
        meta = five_way["meta"]
        if meta.get("degree_shift") == "not_applicable":
            print(f"[OOD_SKIP] method=GFT dataset={data_name} split_seed={split_seed} "
                  f"reason={meta.get('reason', 'degree_shift_not_applicable')} "
                  f"num_classes={meta.get('num_classes')}")
            continue

        print(
            f"[OOD_SPLIT] dataset={data_name} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_20_20_descending')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_deg_range={meta['train_pool_degree_range']} "
            f"ood_val_deg_range={meta['ood_val_degree_range']} "
            f"ood_test_deg_range={meta['ood_test_degree_range']} "
            f"smallest_train_pool_class={meta['smallest_train_pool_class']}"
        )

        # Move split idx tensors to the same device as the graph.
        five_way_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in five_way.items() if k != "meta"
        }

        for run_seed in run_seeds:
            seed_everything(run_seed)

            task_model = TaskModel(
                encoder=deepcopy(encoder),
                vq=deepcopy(vq),
                num_classes=num_classes,
                params=params,
            ).to(device)

            if params.get("freeze_vq", False):
                for p in task_model.vq.parameters():
                    p.requires_grad = False

            opt_params = [p for p in task_model.parameters() if p.requires_grad]
            task_opt = AdamW(opt_params, lr=params["finetune_lr"])
            stopper = EarlyStopping(patience=params["early_stop"])

            # Loader only used if batch_size > 0 (not the case for node configs we care about).
            train_loader = None
            subgraph_loader = None
            if params["batch_size"] != 0:
                train_split_for_loader = _train_split_view(five_way_dev)
                train_loader, subgraph_loader = get_loader(
                    data, train_split_for_loader, labels, params,
                )

            best_id_val = -float("inf")
            best_ood_val = -float("inf")
            best_id_state = None
            best_ood_state = None

            train_view = _train_split_view(five_way_dev)

            for epoch in range(params["finetune_epochs"]):
                loss = ft_node(
                    model=task_model,
                    dataset=data,
                    loader=train_loader,
                    optimizer=task_opt,
                    split=train_view,
                    labels=labels,
                    params=params,
                    num_neighbors=[30] * params["num_layers"],
                )

                metrics = _eval_both(
                    task_model=task_model,
                    data_or_dataset=data,
                    loader=subgraph_loader,
                    five_way=five_way_dev,
                    labels=labels,
                    params=params,
                    num_neighbors=[-1] * params["num_layers"],
                )

                if metrics["id_val"] > best_id_val:
                    best_id_val = metrics["id_val"]
                    best_id_state = {
                        k: v.detach().cpu().clone()
                        for k, v in task_model.state_dict().items()
                    }
                if metrics["ood_val"] > best_ood_val:
                    best_ood_val = metrics["ood_val"]
                    best_ood_state = {
                        k: v.detach().cpu().clone()
                        for k, v in task_model.state_dict().items()
                    }

                # Early stopping follows the main protocol (id_val).
                stopper_input = {
                    "train": metrics["train"],
                    "val": metrics["id_val"],
                    "test": metrics["id_test"],
                    "metric": "acc",
                }
                if stopper(stopper_input):
                    print(f"Early Stopping at Epoch: {epoch}")
                    break

                if (epoch + 1) % 50 == 0 or epoch == 0:
                    print(f"[epoch {epoch:4d}] loss={loss['loss']:.4f} "
                          f"id_val={metrics['id_val']:.2f} ood_val={metrics['ood_val']:.2f} "
                          f"id_test={metrics['id_test']:.2f} ood_test={metrics['ood_test']:.2f}")

                wandb.log({
                    "train/loss": loss["loss"],
                    "train/id_val": metrics["id_val"],
                    "train/ood_val": metrics["ood_val"],
                    "train/id_test": metrics["id_test"],
                    "train/ood_test": metrics["ood_test"],
                })

            # ---- Final eval: reload best-id_val, best-ood_val in turn ----
            def _eval_with_state(state):
                task_model.load_state_dict({k: v.to(device) for k, v in state.items()})
                task_model.eval()
                with torch.no_grad():
                    return _eval_both(
                        task_model=task_model,
                        data_or_dataset=data,
                        loader=subgraph_loader,
                        five_way=five_way_dev,
                        labels=labels,
                        params=params,
                        num_neighbors=[-1] * params["num_layers"],
                    )

            main_m = _eval_with_state(best_id_state) if best_id_state is not None else metrics
            oracle_m = _eval_with_state(best_ood_state) if best_ood_state is not None else metrics

            def _gap(id_v, ood_v):
                gap_abs = id_v - ood_v
                gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
                rr = ood_v / id_v if id_v > 0 else 0.0
                return gap_abs, gap_rel, rr

            gA, gR, rR = _gap(main_m["id_test"], main_m["ood_test"])
            print(f"[OOD_RAW] method=GFT dataset={data_name} split_seed={split_seed} "
                  f"run_seed={run_seed} shift=degree selector=id_val "
                  f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                  f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                  f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}")

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(f"[OOD_ORACLE] method=GFT dataset={data_name} split_seed={split_seed} "
                  f"run_seed={run_seed} shift=degree selector=ood_val "
                  f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                  f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                  f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}")

            all_rows.append({
                "dataset": data_name, "split_seed": split_seed, "run_seed": run_seed,
                "selector": "id_val",
                "id_test": main_m["id_test"], "ood_test": main_m["ood_test"],
                "gap_abs": gA, "gap_rel": gR, "rr": rR,
            })
            all_rows.append({
                "dataset": data_name, "split_seed": split_seed, "run_seed": run_seed,
                "selector": "ood_val",
                "id_test": oracle_m["id_test"], "ood_test": oracle_m["ood_test"],
                "gap_abs": gA_o, "gap_rel": gR_o, "rr": rR_o,
            })

            wandb.log({
                "final/id_test_main": main_m["id_test"],
                "final/ood_test_main": main_m["ood_test"],
                "final/id_test_oracle": oracle_m["id_test"],
                "final/ood_test_oracle": oracle_m["ood_test"],
                "final/gap_abs_main": gA,
                "final/rr_main": rR,
            })

    print("\n=== GFT OOD summary (dataset={}) ===".format(data_name))
    for row in all_rows:
        print("  seed={split_seed}/{run_seed} selector={selector} "
              "id={id_test:.4f} ood={ood_test:.4f} gap={gap_abs:.4f} rr={rr:.4f}".format(**row))

    wandb.finish()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    params = get_args_finetune()

    params["data_path"] = osp.join(osp.dirname(__file__), "..", "data")
    params["pt_model_path"] = osp.join(osp.dirname(__file__), "..", "ckpts", "pretrain_model")

    data_name = params["finetune_dataset"]
    params["task"] = "node"

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), "..", "config", "finetune.yaml"), "r") as f:
            default_params = yaml.safe_load(f)
            if data_name in default_params.get("node", {}):
                cfg = dict(default_params["node"][data_name])
            else:
                # fallback: use cora hyperparams for datasets without a dedicated block
                cfg = dict(default_params["node"]["cora"])
                print(f"[OOD_WARN] no node config for {data_name}, using cora placeholders")
            # Do NOT let the yaml block's own `finetune_dataset` override the
            # CLI dataset (this was a silent bug that caused `gft_products`
            # jobs to actually train cora when products was missing from yaml).
            cfg.pop("finetune_dataset", None)
            params.update(cfg)

    assert not (params["no_lin_clf"] and params["no_proto_clf"])
    if params["no_lin_clf"]:
        params["trade_off"] = 0
    if params["no_proto_clf"]:
        params["trade_off"] = 1

    wandb.init(
        project="GFT-OOD",
        name=f"{str.upper(data_name)} - degree OOD",
        config=params,
        mode="disabled" if params["debug"] else "online",
        tags=["ood", "degree"],
    )
    params = dict(wandb.config)
    print(params)

    run(params)
