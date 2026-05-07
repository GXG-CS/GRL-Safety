#!/usr/bin/env python
# coding: utf-8
"""
GFT fine-tuning + edge-deletion corruption evaluation.

Mirror of `run_feature_noise.py`, but corrupts graph structure at
inference time instead of node features:
- Runs the standard GFT FT training (unchanged).
- After training each split, restores the best-val checkpoint and runs
  a clean eval + 5 edge-drop evals (p ∈ {0.05, 0.10, 0.20, 0.30, 0.50}).
- Appends raw per-(split × severity) rows and emits a single
  `[ED_AGG]` summary line per (method, dataset) for downstream
  aggregation.

Corruption model (per spec
`experiment_design/corruption_edge_deletion/corruption_edge_deletion.md`):
    For each undirected edge (u, v), drop independently with probability p.
    Both directions of the same (min, max) key are dropped/kept together.
    Self-loops are preserved (never dropped).

Only `data.edge_index` (and its paired `data.xe` edge-type index) are
mutated per severity; node features and labels are untouched. The
original edge_index / xe are restored between severities.

GFM-Safety convention: GFT runs with `--no_proto_clf` (linear head only).
The yaml config's `trade_off` would otherwise silently override any CLI
flag, so after yaml loading we force `trade_off=1` when `no_proto_clf`
is set (matches `finetune.py` lines 281-286).
"""

import os
import os.path as osp
import sys
import collections
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.args import get_args_finetune
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.others import seed_everything, load_params, mask2idx
from utils.splitter import get_split, get_split_graph

from task.node import ft_node, eval_node
from task.link import ft_link, eval_link
from task.graph import ft_graph, eval_graph

import warnings
import wandb

warnings.filterwarnings("ignore")

dataset2task = {
    "cora": "node",
    "citeseer": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "elephoto": "node",
    "elecomp": "node",
    # GFM-Safety scope-expansion NC datasets (beyond GFT's upstream set).
    # All use cora-style placeholder hyperparams in config/finetune.yaml.
    "tolokers": "node",
    "dblp": "node",
    "arxiv23": "node",
    "amazonratings": "node",
    "bookhis": "node",
    "bookchild": "node",
    "sportsfit": "node",
    "products": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    # New link datasets (Recommendation + KG-biology expansion, autoSplit via tag_registry).
    "goodreads": "link",
    "ml1m": "link",
    "ml1m_cls": "link",
    "protein_hs": "link",
    "arxivyear": "node",
    "chemhiv": "graph",
    "chempcba": "graph",
    "bace": "graph",
    "bbbp": "graph",
    "cyp450": "graph",
    "muv": "graph",
    "tox21": "graph",
    "toxcast": "graph",
}

# -----------------------------------------------------------------------------
# Edge deletion config (per spec)
# -----------------------------------------------------------------------------

SEVERITIES = [
    (1, 0.05),
    (2, 0.10),
    (3, 0.20),
    (4, 0.30),
    (5, 0.50),
]


class _GraphDataset:
    """Thin wrapper over a list of PyG Data objects that supports tensor indexing."""
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.y = labels
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, (list, np.ndarray)):
            return _GraphDataset([self.graphs[i] for i in idx],
                                 self.y[idx] if self.y is not None else None)
        return self.graphs[idx]
    def __len__(self):
        return len(self.graphs)


def _load_graph_dataset(data_name, project_root):
    """Load graph-classification dataset from TAG .pt + random 80/10/10 split."""
    from torch_geometric.data import Data

    tag_pt = osp.join(project_root, "datasets", "TAG", data_name,
                       "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    edge_text_feat = merged.edge_embs
    class_node_text_feat = merged.class_node_text_feat

    n_graphs = slices["y"].shape[0] - 1
    graphs = []
    for i in range(n_graphs):
        ns = slices["x"][i].item()
        ne = slices["x"][i + 1].item()
        es = slices["edge_index"][i].item()
        ee = slices["edge_index"][i + 1].item()
        atom_idx = merged.x[ns:ne]
        bond_idx = merged.xe[es:ee]
        y_slice = merged.y[slices["y"][i]:slices["y"][i + 1]]
        if y_slice.dim() == 1:
            y_slice = y_slice.unsqueeze(0)
        g = Data(
            x=atom_idx,
            edge_index=merged.edge_index[:, es:ee],
            xe=bond_idx,
            y=y_slice,
            node_text_feat=node_text_feat[atom_idx],
            edge_text_feat=edge_text_feat[bond_idx],
        )
        graphs.append(g)

    labels = merged.y
    y_per_graph = labels.shape[0] // n_graphs
    if y_per_graph == 1:
        num_tasks, num_classes = 1, None
        labels = labels.reshape(-1, 1)
    else:
        num_tasks = y_per_graph
        num_classes = None
        labels = labels.reshape(-1, num_tasks)

    dataset = _GraphDataset(graphs, labels)

    # Random 80/10/10 split (seeded, no OGB dependency)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr = int(0.8 * n_graphs)
    n_va = int(0.1 * n_graphs)
    train_idx = perm[:n_tr]
    val_idx = perm[n_tr:n_tr+n_va]
    test_idx = perm[n_tr+n_va:]

    split = {"train": train_idx, "valid": val_idx, "test": test_idx}
    return dataset, split, labels, num_classes, num_tasks


def apply_edge_drop(
    edge_index: torch.Tensor,
    num_nodes: int,
    p: float,
    drop_seed: int = 0,
):
    """Random Bernoulli edge drop with undirected-pair coupling.

    Each undirected edge (u, v) — identified by canonical key (min, max) —
    is dropped or kept as a single unit, so both directions in a PyG-style
    double-stored undirected graph drop together. Self-loops are preserved.

    Args:
        edge_index: [2, E] edge index tensor.
        num_nodes:  number of nodes N (for the (min, max) key).
        p:          edge-drop probability for each undirected unit.
        drop_seed:  RNG seed for reproducibility.

    Returns:
        (new_edge_index, keep_mask)
            new_edge_index: [2, E'] subset of edges that survive.
            keep_mask:      [E] bool, same mask to apply to any per-edge
                             tensor paired with `edge_index` (e.g. `data.xe`).
    """
    device = edge_index.device
    E = edge_index.size(1)

    if p <= 0.0 or E == 0:
        keep = torch.ones(E, dtype=torch.bool, device=device)
        return edge_index, keep

    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    key = u.long() * int(num_nodes) + v.long()

    _, inverse = torch.unique(key, return_inverse=True)
    num_undirected = int(inverse.max().item()) + 1

    g = torch.Generator(device=device).manual_seed(int(drop_seed))
    undirected_keep = torch.rand(num_undirected, generator=g, device=device) >= p
    keep = undirected_keep[inverse]
    keep = keep | (src == dst)  # preserve self-loops

    return edge_index[:, keep], keep


# -----------------------------------------------------------------------------
# Task dispatch (identical to finetune.py)
# -----------------------------------------------------------------------------

def get_preprocess(params):
    if params['task'] == 'node':
        return pre_node
    elif params['task'] == 'link':
        return pre_link
    elif params['task'] == 'graph':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')


def get_ft(params):
    task = params['task']

    if task == "node":
        return ft_node
    elif task == "link":
        return ft_link
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Invalid Task")


def get_eval(params):
    task = params['task']

    if task == "node":
        return eval_node
    elif task == "link":
        return eval_link
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Invalid Task")


# -----------------------------------------------------------------------------
# Main run loop: FT (unchanged) + edge-deletion eval (new)
# -----------------------------------------------------------------------------

def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    preprocess = get_preprocess(params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

    data_name = params["finetune_dataset"]
    task = params["task"]
    setting = params["setting"]

    # Load data via tag_registry (auto-prepare if not cached)
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag
    data, _ = load_tag(data_name)

    # Ensure x is 1D index and xe exists (GFT original: span_node_and_edge_idx)
    if data.x is None or data.x.ndim == 2:
        data.x = torch.arange(data.node_text_feat.size(0), dtype=torch.long)
    if not hasattr(data, 'xe') or data.get('xe') is None:
        data.xe = torch.zeros(data.edge_index.size(1), dtype=torch.long)

    # Labels and splits depend on task type (same logic as original get_finetune_graph)
    if task == "link":
        if hasattr(data, 'edge_types') and data.edge_types is not None:
            labels = data.edge_types
            num_classes = int(labels.unique().shape[0])
        else:
            labels = torch.zeros(data.edge_index.size(1), dtype=torch.long)
            num_classes = 2
        n_train = len(data.train_idx)
        n_valid = len(data.val_idx)
        n_test  = len(data.test_idx)
        splits = [{"train": torch.arange(0, n_train),
                   "valid": torch.arange(n_train, n_train + n_valid),
                   "test":  torch.arange(n_train + n_valid, n_train + n_valid + n_test)}]
    elif task == "node":
        labels = data.y.squeeze()
        num_classes = int(labels.max().item()) + 1
        splits = []
        if hasattr(data, 'train_masks'):
            avail = len(data.train_masks)
            for i in range(5):
                j = i % avail
                splits.append({"train": data.train_masks[j],
                               "valid": data.val_masks[j],
                               "test": data.test_masks[j]})
        else:
            tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
            if tm.dim() == 2:
                avail = tm.size(1)
                for i in range(5):
                    j = i % avail
                    test_i = tsm[:, j] if tsm.dim() == 2 else tsm
                    splits.append({"train": tm[:, j], "valid": vm[:, j], "test": test_i})
            else:
                splits = [{"train": tm, "valid": vm, "test": tsm}] * 5
    elif task == "graph":
        graph_dataset, graph_split, labels, num_classes, num_tasks = _load_graph_dataset(data_name, _project_root)
        num_classes = num_tasks
        splits = [graph_split] * params["repeat"]
        dataset = graph_dataset
    else:
        raise ValueError(f"Unknown task: {task}")

    if task != "graph":
        num_tasks = 1
    params["num_classes"] = num_classes
    data.y = labels

    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

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

    # Load Pretrained Model
    # Current GFM-Safety stores GFT checkpoints in a flat repo-level layout:
    #   <repo>/ckpts/GFT/{encoder.pt,vq.pt}
    # Keep `ckpt_dir` as an optional override, but otherwise treat any
    # non-'na' pretrain_dataset as "load the default pretrained pair".
    ckpt_dir = params.get("ckpt_dir")
    if ckpt_dir is None and params["pretrain_dataset"] != 'na':
        ckpt_dir = osp.abspath(osp.join(
            osp.dirname(__file__), "..", "..", "..", "..", "ckpts", "GFT"
        ))

    if ckpt_dir:
        encoder = load_params(encoder, osp.join(ckpt_dir, "encoder.pt"))
        vq = load_params(vq, osp.join(ckpt_dir, "vq.pt"))
        print("Loaded pretrained encoder and vq from {}".format(ckpt_dir))

    train_loader = None
    val_loader = None
    test_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()

    seeds = params.get("seeds", [42, 43, 44, 45, 46])
    n_runs = len(seeds)

    if len(splits) >= n_runs:
        run_configs = [(seeds[i], deepcopy(splits[i])) for i in range(n_runs)]
    else:
        run_configs = [(s, deepcopy(splits[0])) for s in seeds]

    # Optional: restrict to a single (split_idx, seed) pair via --split_idx arg.
    only_idx = params.get("split_idx", None)
    if only_idx is not None and only_idx >= 0:
        run_configs = [run_configs[only_idx]]

    # Holds one dict per (split, severity) including clean (sev=0)
    all_results = []

    for idx, (seed, split) in enumerate(run_configs):
        seed_everything(seed)

        if setting == "standard":
            split = split
        elif setting in ["few_shot", "zero_shot", "in_context"]:
            if task in ["node", "link"]:
                split = get_split(split, labels, params)
            elif task == "graph":
                split = get_split_graph(split, labels, params)
        else:
            raise ValueError("Invalid Setting")

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

        if params["batch_size"] != 0 and task in ["node", "link"]:
            train_loader, subgraph_loader = get_loader(data, split, labels, params)
        elif task == "graph":
            if params["batch_size"] == 0:
                params["batch_size"] = 256
            train_loader, val_loader, test_loader = get_loader(dataset, split, labels, params)

        # NEW: track best-val checkpoint so corruption eval uses the same
        # model state that produced the reported clean accuracy.
        best_val = -float("inf")
        best_state = None

        for epoch in range(params["finetune_epochs"]):
            loss = finetune(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=train_loader,
                optimizer=task_opt,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[30] * params["num_layers"],
            )

            result = evaluate(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=subgraph_loader if task in ["node", "link"] else [train_loader, val_loader, test_loader],
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[-1] * params["num_layers"],
            )

            # NEW: save state if val improved
            if result["val"] > best_val:
                best_val = result["val"]
                best_state = {k: v.detach().cpu().clone() for k, v in task_model.state_dict().items()}

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            wandb.log(
                {
                    "train/proto_loss": loss['proto_loss'],
                    "train/lin_loss": loss['act_loss'],
                    "train/loss": loss['loss'],
                    "train/train_value": result['train'],
                    "train/val_value": result['val'],
                    "train/test_value": result['test'],
                }
            )

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })

        # -------- NEW: corruption eval on best-val checkpoint --------
        if best_state is not None:
            task_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        task_model.eval()

        if task == "graph":
            # ----- Graph-classification edge deletion eval -----
            from torch_geometric.loader import DataLoader as PyGDataLoader
            from torch_geometric.data import Data as PyGData

            test_graphs = dataset[split["test"]]

            # Clean eval
            with torch.no_grad():
                clean_result = evaluate(
                    model=task_model, dataset=dataset,
                    loader=[train_loader, val_loader, test_loader],
                    split=split, labels=labels, params=params,
                    num_neighbors=[-1] * params["num_layers"],
                )
            clean_acc = float(clean_result["test"])
            all_results.append({"split_idx": idx, "seed": seed, "sev": 0,
                                "p": 0.0, "test_acc": clean_acc})
            print(f"[ED_RAW] method=GFT dataset={data_name} "
                  f"split_idx={idx} seed={seed} sev=0 p=0.0 "
                  f"test_acc={clean_acc:.4f}")

            for sev_idx, p_drop in SEVERITIES:
                noisy_test = []
                for gi, g in enumerate(test_graphs):
                    gc = g.clone()
                    new_ei, keep = apply_edge_drop(
                        gc.edge_index, num_nodes=gc.x.size(0), p=p_drop,
                        drop_seed=seed * 100000 + sev_idx * 10000 + gi,
                    )
                    gc.edge_index = new_ei
                    gc.edge_text_feat = gc.edge_text_feat[keep]
                    gc.xe = gc.xe[keep]
                    noisy_test.append(gc)
                noisy_loader = PyGDataLoader(noisy_test, batch_size=params["batch_size"],
                                              shuffle=False, num_workers=8)
                with torch.no_grad():
                    noisy_result = evaluate(
                        model=task_model, dataset=dataset,
                        loader=[train_loader, val_loader, noisy_loader],
                        split=split, labels=labels, params=params,
                        num_neighbors=[-1] * params["num_layers"],
                    )
                noisy_acc = float(noisy_result["test"])
                all_results.append({"split_idx": idx, "seed": seed, "sev": sev_idx,
                                    "p": p_drop, "test_acc": noisy_acc})
                print(f"[ED_RAW] method=GFT dataset={data_name} "
                      f"split_idx={idx} seed={seed} sev={sev_idx} p={p_drop} "
                      f"test_acc={noisy_acc:.4f}")
        else:
            # ----- Node / Link corruption eval -----
            original_edge_index = data.edge_index.clone()
            original_xe = data.xe.clone()
            num_nodes_N = int(data.node_text_feat.size(0))

            # Clean eval
            with torch.no_grad():
                clean_result = evaluate(
                    model=task_model,
                    dataset=data, loader=subgraph_loader,
                    split=split, labels=labels, params=params,
                    num_neighbors=[-1] * params["num_layers"],
                )

            clean_acc = float(clean_result["test"])
            clean_f1 = float(clean_result.get("test_f1", 0.0))
            clean_auc = float(clean_result.get("test_auc", 0.0))
            all_results.append({"split_idx": idx, "seed": seed, "sev": 0,
                                "p": 0.0, "test_acc": clean_acc,
                                "macro_f1": clean_f1, "test_auc": clean_auc})
            print(f"[ED_RAW] method=GFT dataset={data_name} "
                  f"split_idx={idx} seed={seed} sev=0 p=0.0 "
                  f"test_acc={clean_acc:.4f} macro_f1={clean_f1:.4f} "
                  f"test_auc={clean_auc:.4f}")

            for sev_idx, p_drop in SEVERITIES:
                if task == "link":
                    n_train = split["train"].size(0) if isinstance(split["train"], torch.Tensor) else len(split["train"])
                    n_val = split["valid"].size(0) if isinstance(split["valid"], torch.Tensor) else len(split["valid"])
                    n_test = split["test"].size(0) if isinstance(split["test"], torch.Tensor) else len(split["test"])
                    train_ei = original_edge_index[:, :n_train]
                    train_xe = original_xe[:n_train]
                    rest_ei = original_edge_index[:, n_train:]
                    rest_xe = original_xe[n_train:]
                    dropped_ei, keep_train = apply_edge_drop(
                        train_ei, num_nodes=num_nodes_N, p=p_drop,
                        drop_seed=seed * 100 + sev_idx,
                    )
                    n_train_new = int(keep_train.sum().item())
                    data.edge_index = torch.cat([dropped_ei, rest_ei], dim=1)
                    data.xe = torch.cat([train_xe[keep_train], rest_xe])
                    # Also filter labels to match surviving edges
                    original_labels = labels
                    train_labels = labels[:n_train]
                    rest_labels = labels[n_train:]
                    labels = torch.cat([train_labels[keep_train], rest_labels])
                    eval_split = {
                        "train": torch.arange(0, n_train_new, device=data.edge_index.device),
                        "valid": torch.arange(n_train_new, n_train_new + n_val, device=data.edge_index.device),
                        "test": torch.arange(n_train_new + n_val, n_train_new + n_val + n_test, device=data.edge_index.device),
                    }
                else:
                    new_ei, keep = apply_edge_drop(
                        original_edge_index, num_nodes=num_nodes_N, p=p_drop,
                        drop_seed=seed * 100 + sev_idx,
                    )
                    data.edge_index = new_ei
                    data.xe = original_xe[keep]
                    eval_split = split

                with torch.no_grad():
                    noisy_result = evaluate(
                        model=task_model, dataset=data, loader=subgraph_loader,
                        split=eval_split, labels=labels, params=params,
                        num_neighbors=[-1] * params["num_layers"],
                    )
                noisy_acc = float(noisy_result["test"])
                noisy_f1 = float(noisy_result.get("test_f1", 0.0))
                noisy_auc = float(noisy_result.get("test_auc", 0.0))
                all_results.append({"split_idx": idx, "seed": seed, "sev": sev_idx,
                                    "p": p_drop, "test_acc": noisy_acc,
                                    "macro_f1": noisy_f1, "test_auc": noisy_auc})
                print(f"[ED_RAW] method=GFT dataset={data_name} "
                      f"split_idx={idx} seed={seed} sev={sev_idx} p={p_drop} "
                      f"test_acc={noisy_acc:.4f} macro_f1={noisy_f1:.4f} "
                      f"test_auc={noisy_auc:.4f}")
                data.edge_index = original_edge_index
                data.xe = original_xe
                if task == "link":
                    labels = original_labels

            data.edge_index = original_edge_index
            data.xe = original_xe
        # -------- end corruption eval block --------

    best = logger.get_best()

    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    print(f"\n=== GFT FT Result (clean, best-val, from logger) ===")
    print(f"Train: {best['train']['mean']:.2f} +/- {best['train']['std']:.2f}")
    print(f"Val:   {best['val']['mean']:.2f} +/- {best['val']['std']:.2f}")
    print(f"Test:  {best['test']['mean']:.2f} +/- {best['test']['std']:.2f}")

    # NEW: aggregate results per severity (mean ± std across splits) for printing
    print("\n=== GFT Edge Deletion Results (aggregated over splits) ===")
    grouped_acc = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    grouped_auc = collections.defaultdict(list)
    for row in all_results:
        grouped_acc[row["sev"]].append(row["test_acc"])
        grouped_f1[row["sev"]].append(row.get("macro_f1", 0.0))
        grouped_auc[row["sev"]].append(row.get("test_auc", 0.0))

    label_for_sev = {0: "clean   "}
    for sev_idx, p_drop in SEVERITIES:
        label_for_sev[sev_idx] = f"sev{sev_idx} p={p_drop}"

    agg_acc, agg_f1, agg_auc = {}, {}, {}
    for sev in sorted(grouped_acc.keys()):
        accs = np.array(grouped_acc[sev], dtype=np.float64)
        f1s = np.array(grouped_f1[sev], dtype=np.float64)
        aucs = np.array(grouped_auc[sev], dtype=np.float64)
        agg_acc[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
        agg_f1[sev] = f"{f1s.mean():.2f} ± {f1s.std():.2f}"
        agg_auc[sev] = f"{aucs.mean():.2f} ± {aucs.std():.2f}"
        print(f"  {label_for_sev[sev]:<14}  acc={agg_acc[sev]}  f1={agg_f1[sev]}  auc={agg_auc[sev]}")

    # Single structured line so an aggregator script can grep it later.
    print(
        f"[ED_AGG] method=GFT dataset={data_name} "
        f"clean=\"{agg_acc.get(0, '')}\" "
        f"sev1=\"{agg_acc.get(1, '')}\" "
        f"sev2=\"{agg_acc.get(2, '')}\" "
        f"sev3=\"{agg_acc.get(3, '')}\" "
        f"sev4=\"{agg_acc.get(4, '')}\" "
        f"sev5=\"{agg_acc.get(5, '')}\" "
        f"clean_f1=\"{agg_f1.get(0, '')}\" "
        f"sev1_f1=\"{agg_f1.get(1, '')}\" "
        f"sev2_f1=\"{agg_f1.get(2, '')}\" "
        f"sev3_f1=\"{agg_f1.get(3, '')}\" "
        f"sev4_f1=\"{agg_f1.get(4, '')}\" "
        f"sev5_f1=\"{agg_f1.get(5, '')}\" "
        f"clean_auc=\"{agg_auc.get(0, '')}\" "
        f"sev1_auc=\"{agg_auc.get(1, '')}\" "
        f"sev2_auc=\"{agg_auc.get(2, '')}\" "
        f"sev3_auc=\"{agg_auc.get(3, '')}\" "
        f"sev4_auc=\"{agg_auc.get(4, '')}\" "
        f"sev5_auc=\"{agg_auc.get(5, '')}\""
    )

    wandb.finish()


if __name__ == "__main__":
    params = get_args_finetune()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pt_model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    dataset = params["finetune_dataset"]
    task = dataset2task[dataset]
    params['task'] = task

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'finetune.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            if task in default_params and dataset in default_params[task]:
                params.update(default_params[task][dataset])

    if params["setting"] in ["few_shot"]:
        if params['finetune_dataset'] in ['FB15K237']:
            params['batch_size'] = 0
        if task == 'graph':
            params['n_way'] = 2
            params['num_instances_per_class'] = params['n_train']

    # Mirror finetune.py: at least one classifier must be enabled, and
    # disabling one forces trade_off to the other side. GFM-Safety uses
    # linear-only (no proto), so --no_proto_clf at the CLI forces trade_off=1
    # *after* yaml loading (the yaml's per-dataset trade_off would otherwise
    # silently override the CLI flag).
    assert not (params['no_lin_clf'] and params['no_proto_clf'])
    if params['no_lin_clf']:
        params['trade_off'] = 0
    if params['no_proto_clf']:
        params['trade_off'] = 1

    wandb.init(
        project="GFT-Finetune-EdgeDeletion",
        name="{} - FT+ED".format(str.upper(params["finetune_dataset"])),
        config=params,
        mode=params.get("wandb_mode", "offline"),
        tags=[params['setting'], "edge_deletion"],
    )
    params = dict(wandb.config)
    print(params)
    print(f"[Self-check] no_proto_clf={params['no_proto_clf']}, trade_off={params['trade_off']}")

    run(params)
