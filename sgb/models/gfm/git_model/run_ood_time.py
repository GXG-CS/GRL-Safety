"""GIT FT + node-level OOD evaluation (GFM-Safety Dim 2, node only).

Degree-based covariate shift following GOOD's 10/1/1 protocol. See
`experiment_design/ood/ood_experiment_design.md` for the full spec.

This file is a node-only fork of `run_edge_deletion.py`. It replaces
GIT's default split (loaded by `utils.split.get_split`) with a frozen,
shared degree-split artifact produced by `sgb.data.ood_splits`, trains on
the `train` pool with `id_val` as the selector (main protocol), and also
tracks the `ood_val`-best checkpoint (appendix oracle). It emits
`[OOD_RAW]` / `[OOD_ORACLE]` log lines consistent with the other
run_ood_degree.py files across the repo.
"""

import os
import os.path as osp
import sys
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data.pretrain_data import domain2task, dataset2domain
from model.encoder import Encoder
from model.finetune_model import TaskModel
from utils.utils import seed_everything, load_params
from utils.args import get_args_finetune
from utils.early_stop import EarlyStopping

from task.node import ft_node, eval_node

import wandb
import warnings

warnings.filterwarnings("ignore")


NODE_DATASETS = {"arxiv"}  # time shift only available on arxiv

SPLIT_SEEDS_DEFAULT = [0]
RUN_SEEDS_DEFAULT = [42, 43, 44, 45, 46]


# -----------------------------------------------------------------------------
# Degree split builder (GOOD 60/20/20 descending, inlined per-method)
# -----------------------------------------------------------------------------
#
# Self-contained copy. Matches GOOD.data.good_datasets.good_cora
# .get_covariate_shift_graph: sort descending by degree, 60/20/20 slice,
# then random-shuffle (seeded) the train slice to carve id_val / id_test
# (each 10% of total). No disk caching.

def _compute_node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Undirected-style node degree (each edge contributes to both endpoints)."""
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg



def _train_split_view(five_way):
    """Train-time split dict fed to ft_node / eval_node (GIT uses 'val' key)."""
    return {
        "train": five_way["train"],
        "val": five_way["id_val"],
        "test": five_way["id_test"],
    }


def _ood_split_view(five_way):
    """Split dict whose 'val'/'test' are the OOD pools (same 'train' for proto)."""
    return {
        "train": five_way["train"],
        "val": five_way["ood_val"],
        "test": five_way["ood_test"],
    }


def _eval_both(task_model, data, five_way, params):
    id_view = _train_split_view(five_way)
    ood_view = _ood_split_view(five_way)
    with torch.no_grad():
        r_id = eval_node(model=task_model, data=data, split=id_view, params=params)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        r_ood = eval_node(model=task_model, data=data, split=ood_view, params=params)
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


def _gap(id_v, ood_v):
    if id_v is None or ood_v is None or id_v != id_v or ood_v != ood_v:
        return (float("nan"),) * 3
    gap_abs = id_v - ood_v
    gap_rel = gap_abs / id_v * 100.0 if id_v > 0 else 0.0
    rr = ood_v / id_v if id_v > 0 else 0.0
    return gap_abs, gap_rel, rr


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data_name = params["dataset"]
    if data_name not in NODE_DATASETS:
        raise ValueError(
            f"GIT run_ood_degree.py currently supports node datasets only; "
            f"got {data_name}. Supported: {sorted(NODE_DATASETS)}"
        )
    params["task"] = "node"

    # ---- Load data via tag_registry (shared interface) ----
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag
    from sgb.data.ood_splits import build_time_shift_split

    graph, _ = load_tag(data_name)
    graph.name = data_name
    if graph.y is None and hasattr(graph, "edge_types") and graph.edge_types is not None:
        graph.y = graph.edge_types
    if graph.y.dim() > 1:
        graph.y = graph.y.squeeze()
    graph.num_classes = int(graph.y.max().item()) + 1
    graph.num_nodes = graph.node_text_feat.size(0)
    graph.num_edges = graph.edge_index.size(1)
    print(
        f"[GIT FT-OOD] Dataset: {graph.name}, #Nodes: {graph.num_nodes}, "
        f"#Edges: {graph.num_edges}, #Classes: {graph.num_classes}"
    )

    # ---- Encoder + pretrained ckpt ----
    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )

    ckpt_dir = params.get("ckpt_dir")
    if ckpt_dir:
        path = osp.join(ckpt_dir, "encoder.pt")
        encoder = load_params(encoder, path)
        print(f"[GIT FT-OOD] Loaded pretrained encoder from {ckpt_dir}")
    elif params.get("pt_data", "na") != "na":
        # GFM-Safety stores a flat GIT ckpt at ckpts/GIT/all/encoder.pt (or similar)
        flat_dir = osp.abspath(osp.join(_project_root, "ckpts", "GIT", "all"))
        flat_path = osp.join(flat_dir, "encoder.pt")
        if osp.exists(flat_path):
            encoder = load_params(encoder, flat_path)
            print(f"[GIT FT-OOD] Loaded pretrained encoder from {flat_dir}")
        else:
            print(f"[GIT FT-OOD] WARNING: no pretrained encoder found at {flat_path}")

    model = TaskModel(encoder, num_classes=graph.num_classes).to(device)

    # ---- Seeds ----
    split_seeds = list(params.get("split_seeds", SPLIT_SEEDS_DEFAULT))
    run_seeds = list(params.get("seeds", RUN_SEEDS_DEFAULT))
    if params.get("debug", False):
        split_seeds = split_seeds[:1]
        run_seeds = run_seeds[:1]
        print(f"[OOD_SMOKE] debug mode: split_seeds={split_seeds} run_seeds={run_seeds}")

    if params["bs"] != 0:
        print(f"[GIT FT-OOD] NOTE: bs={params['bs']} (using loader). node batching path untested for OOD; falling back to full-batch.")
        params["bs"] = 0

    # Move full-graph data to device once
    data = deepcopy(graph).to(device)
    labels = data.y

    # Time-shift: require node_year
    if not hasattr(graph, "node_year") or graph.node_year is None:
        raise RuntimeError(f"Dataset has no node_year; time shift N/A.")
    year_t = graph.node_year
    
    for split_seed in split_seeds:
        five_way = build_time_shift_split(
            dataset_name=data_name,
            year_tensor=year_t,
            labels=labels,
            split_seed=split_seed,
            train_max_year=params.get("train_max_year") or None,
            ood_min_year=params.get("ood_min_year") or None
        )
        meta = five_way["meta"]
        if meta.get("time_shift") == "not_applicable":
            print(f"[OOD_SKIP] method=GIT dataset={data_name} split_seed={split_seed} "
                  f"reason={meta.get('reason', 'unknown')} num_classes={meta.get('num_classes')}")
            continue

        print(
            f"[OOD_SPLIT] dataset={data_name} split_seed={split_seed} "
            f"strategy={meta.get('strategy', 'good_60_5_35_ascending_year')} "
            f"train_pool={meta['train_pool_size']} actual_train={meta['actual_train_size']} "
            f"id_val={meta['id_val_size']} id_test={meta['id_test_size']} "
            f"ood_val={meta['ood_val_size']} ood_test={meta['ood_test_size']} "
            f"train_year_range={meta['train_pool_year_range']} "
            f"ood_val_year_range={meta['ood_val_year_range']} "
            f"ood_test_year_range={meta['ood_test_year_range']} "
            f"smallest_train_pool_class={meta['smallest_train_pool_class']}"
        )

        # Device-resident idx tensors
        five_way_dev = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in five_way.items() if k != "meta"
        }

        for run_seed in run_seeds:
            seed_everything(run_seed)

            task_model = deepcopy(model).to(device)
            optimizer = AdamW(
                task_model.parameters(),
                lr=params["lr"],
                weight_decay=params["decay"],
            )
            stopper = EarlyStopping(patience=params["early_stop"])

            best_id_val = -float("inf")
            best_ood_val = -float("inf")
            best_id_state = None
            best_ood_state = None

            train_view = _train_split_view(five_way_dev)

            for epoch in range(1, params["epochs"] + 1):
                loss = ft_node(
                    model=task_model, data=data, split=train_view,
                    optimizer=optimizer, params=params,
                )
                metrics = _eval_both(task_model, data, five_way_dev, params)

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

                stopper_input = {
                    "train": metrics["train"],
                    "val": metrics["id_val"],
                    "test": metrics["id_test"],
                    "metric": "acc",
                }
                if stopper(stopper_input):
                    print(f"Early Stopping at Epoch: {epoch}")
                    break

                if epoch == 1 or epoch % 50 == 0:
                    print(
                        f"[epoch {epoch:4d}] loss={loss if isinstance(loss, float) else float(loss):.4f} "
                        f"id_val={metrics['id_val']:.2f} ood_val={metrics['ood_val']:.2f} "
                        f"id_test={metrics['id_test']:.2f} ood_test={metrics['ood_test']:.2f}"
                    )

                wandb.log({
                    "train/id_val": metrics["id_val"],
                    "train/ood_val": metrics["ood_val"],
                    "train/id_test": metrics["id_test"],
                    "train/ood_test": metrics["ood_test"],
                })

            def _eval_with_state(state):
                if state is None:
                    return None
                task_model.load_state_dict({k: v.to(device) for k, v in state.items()})
                task_model.eval()
                with torch.no_grad():
                    return _eval_both(task_model, data, five_way_dev, params)

            main_m = _eval_with_state(best_id_state)
            oracle_m = _eval_with_state(best_ood_state)
            if main_m is None:
                main_m = {"id_test": float("nan"), "ood_test": float("nan"),
                          "id_val": 0.0, "ood_val": 0.0}
            if oracle_m is None:
                oracle_m = main_m

            gA, gR, rR = _gap(main_m["id_test"], main_m["ood_test"])
            print(
                f"[OOD_RAW] method=GIT dataset={data_name} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=time selector=id_val "
                f"id={main_m['id_test']:.4f} ood={main_m['ood_test']:.4f} "
                f"gap_abs={gA:.4f} gap_rel={gR:.4f} rr={rR:.4f} "
                f"id_val={main_m['id_val']:.4f} ood_val={main_m['ood_val']:.4f}"
            )

            gA_o, gR_o, rR_o = _gap(oracle_m["id_test"], oracle_m["ood_test"])
            print(
                f"[OOD_ORACLE] method=GIT dataset={data_name} "
                f"split_seed={split_seed} run_seed={run_seed} "
                f"shift=time selector=ood_val "
                f"id={oracle_m['id_test']:.4f} ood={oracle_m['ood_test']:.4f} "
                f"gap_abs={gA_o:.4f} gap_rel={gR_o:.4f} rr={rR_o:.4f} "
                f"id_val={oracle_m['id_val']:.4f} ood_val={oracle_m['ood_val']:.4f}"
            )


def main():
    params = get_args_finetune()
    params["train_max_year"] = 2010
    params["ood_min_year"] = 2017
    params["data_path"] = osp.join(os.path.dirname(__file__), "cache_data")
    params["pt_model_path"] = osp.join(os.path.dirname(__file__), "model", "pretrain_model")
    params["sft_model_path"] = osp.join(os.path.dirname(__file__), "model", "sft_model")
    params["ft_model_path"] = osp.join(os.path.dirname(__file__), "model", "finetune_model")

    dataset = params["dataset"]
    params["task"] = "node"

    if params["use_params"]:
        config_path = osp.join(osp.dirname(__file__), "config", f"{params['setting']}.yaml")
        with open(config_path, "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params.get("base", {}))
            node_cfg = default_params.get("node", {})
            if dataset in node_cfg:
                params.update(node_cfg[dataset])
            else:
                # fall back to cora placeholders if no per-dataset config exists
                if "cora" in node_cfg:
                    params.update(node_cfg["cora"])
                    print(f"[GIT FT-OOD] No node config for {dataset}, using cora placeholders")

    tags = ["node", params["setting"], "ood_degree"]
    wandb.init(
        project="GIT-Finetune-OOD",
        name=f"Data:{dataset} | degree OOD",
        config=params,
        mode="disabled" if params.get("debug", False) else params.get("wandb_mode", "offline"),
        tags=tags,
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
