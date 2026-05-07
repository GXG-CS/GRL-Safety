"""Shared graph-level interpretation FT runner.

Parallel to sgb/data/scaffold_ood_runner.py. Each method-specific
run_interpretation_graph.py provides a model factory and calls
run_graph_interpretation(...). The runner:
  - Builds per-graph PyG Data (TAG SBERT features) for the chosen dataset
  - Loads scaffold split (BACE/Tox21) OR random 80/10/10 (MUTAG)
  - For each seed: build a fresh model, FT to convergence, then run
    `run_graph_fidelity` from sgb.metrics.interpretation_graph
  - Emits [GINTERP_V2_RAW] log lines + [GINTERP_V2_AGG] summary
"""
from __future__ import annotations

import copy
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from sgb.data.scaffold_ood_runner import _build_graphs, _compute_auc
from sgb.metrics.interpretation_graph import (
    run_graph_fidelity,
    format_raw_log,
)


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))


def _stratified_random_split(n: int, y: np.ndarray, seed: int,
                             frac=(0.8, 0.1, 0.1)):
    """Stratified random split for binary y, returns dict of np arrays."""
    rng = np.random.RandomState(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(round(frac[0] * len(idx)))
        n_va = int(round(frac[1] * len(idx)))
        splits['train'].extend(idx[:n_tr].tolist())
        splits['val'].extend(idx[n_tr:n_tr + n_va].tolist())
        splits['test'].extend(idx[n_tr + n_va:].tolist())
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def _build_graphs_ba2motifs():
    """Build BA-2Motifs from PyG, project node features to 768d via fixed
    random projection. BA-2Motifs has 1000 graphs, each with a 20-node BA
    base (atom indices 0..19) plus a 5-node motif (indices 20..24).
    Class 0 = House, Class 1 = 5-Cycle. Motif atoms = indices >= 20.
    """
    from torch_geometric.datasets import BA2MotifDataset
    from torch_geometric.data import Data
    root = osp.join(_PROJECT_ROOT, "datasets", "ba2motifs")
    ds = BA2MotifDataset(root=root)
    rng = np.random.RandomState(0)
    in_dim = ds[0].x.size(1) if ds[0].x is not None else 10
    proj = torch.from_numpy(
        (rng.randn(in_dim, 768) / np.sqrt(in_dim)).astype(np.float32)
    )
    graphs = []
    ys = []
    motif_masks = {}
    for gi, data in enumerate(ds):
        x_in = data.x.float() if data.x is not None else \
               torch.eye(data.num_nodes)[:, :in_dim].float()
        x_proj = x_in @ proj
        # motif mask: atoms with index >= 20 (BA base is 0..19)
        n = x_proj.size(0)
        mmask = torch.zeros(n, dtype=torch.bool)
        if n > 20:
            mmask[20:] = True
        y_int = int(data.y.item()) if data.y.numel() == 1 else int(data.y.view(-1)[0].item())
        new = Data(
            x=x_proj,
            edge_index=data.edge_index,
            y=torch.tensor([y_int], dtype=torch.long),
        )
        graphs.append(new)
        ys.append(y_int)
        motif_masks[gi] = mmask
    return graphs, 1, len(graphs), np.array(ys), motif_masks


def run_graph_interpretation(
    *,
    method_tag: str,
    dataset: str,
    build_ft_model: Callable[[int, int, float, torch.device], nn.Module],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.2,
    max_epochs: int = 500,
    patience: int = 200,
    n_seeds: int = 5,
    batch_size: int = 256,
    topk_fracs=(0.2, 0.3),
    device: torch.device = None,
    motif_masks: Optional[dict] = None,
    saliency_task_idx: int = 0,
):
    """FT then graph-level interpretation. Aligned with scaffold-OOD FT
    setup so accuracy is comparable to scaffold-OOD numbers.

    For dataset in {bace, tox21}: uses scaffold split (test set = scaffold OOD).
    For dataset == mutag: stratified random 80/10/10.
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    device = device or (torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu"))
    print(f"[{method_tag} GINTERP] device={device} dataset={dataset}")

    # Build graphs + splits
    bm_motif_masks = None
    # All TAG-cached mol graph-cls datasets use the same loader.
    TAG_MOL_DATASETS = {"bbbp", "bace", "tox21", "chemhiv", "cyp450",
                        "muv", "chempcba"}
    if dataset in TAG_MOL_DATASETS:
        graphs, num_tasks, n_graphs = _build_graphs(dataset)
        # Stratified random 80/10/10 by task-0 label (drop NaN-task-0 graphs).
        ys_for_split = []
        valid_idx = []
        for i, g in enumerate(graphs):
            y_full = g.y.view(-1)
            if num_tasks > 1:
                y0 = y_full[saliency_task_idx]
            else:
                y0 = y_full[0]
            if torch.isnan(y0):
                continue
            ys_for_split.append(int(y0.item()))
            valid_idx.append(i)
        ys_for_split = np.array(ys_for_split)
        sp = _stratified_random_split(len(valid_idx), ys_for_split, seed=42)
        train_idx = [valid_idx[i] for i in sp['train']]
        val_idx = [valid_idx[i] for i in sp['val']]
        test_idx = [valid_idx[i] for i in sp['test']]
        split_tag = "random"
    elif dataset == "ba2motifs":
        graphs, num_tasks, n_graphs, ys, bm_motif_masks = _build_graphs_ba2motifs()
        sp = _stratified_random_split(n_graphs, ys, seed=42)
        train_idx, val_idx, test_idx = sp['train'], sp['val'], sp['test']
        split_tag = "random"
        if motif_masks is None:
            motif_masks = bm_motif_masks
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    print(f"[{method_tag} GINTERP] n_graphs={n_graphs} num_tasks={num_tasks} "
          f"split={split_tag} train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    is_multitask = num_tasks > 1

    all_aggs = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader = PyGDataLoader([graphs[i] for i in train_idx],
                                     batch_size=batch_size, shuffle=True)
        val_loader = PyGDataLoader([graphs[i] for i in val_idx],
                                   batch_size=512, shuffle=False)
        test_loader = PyGDataLoader([graphs[i] for i in test_idx],
                                    batch_size=512, shuffle=False)

        in_ch = graphs[0].x.size(1)
        model = build_ft_model(in_ch, num_tasks, dropout, device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

        best_val, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optim.zero_grad()
                logits = model(batch)
                if is_multitask:
                    targets = batch.y.float().view(-1, num_tasks)
                    mask = ~torch.isnan(targets)
                    loss = F.binary_cross_entropy_with_logits(
                        logits[mask], targets[mask])
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), batch.y.float())
                loss.backward()
                optim.step()
            model.eval()
            val_auc, _ = _compute_auc(val_loader, model, num_tasks, device)
            if val_auc > best_val:
                best_val = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        test_auc, test_f1 = _compute_auc(test_loader, model, num_tasks, device)
        print(f"[GINTERP_FT] method={method_tag} dataset={dataset} "
              f"seed={seed} val_auc={best_val:.4f} test_auc={test_auc:.4f}")

        # Now run graph-level fidelity on test set, single-graph at a time
        # For multi-task: project y to scalar via saliency_task_idx, drop NaN
        from torch_geometric.data import Data as PyGData
        test_graphs = []
        test_orig_idx = []  # original positions in graphs
        for orig in test_idx:
            g = graphs[orig]
            if is_multitask:
                y_full = g.y.view(-1)
                if saliency_task_idx >= y_full.numel():
                    continue
                y_v = y_full[saliency_task_idx]
                if torch.isnan(y_v):
                    continue
                yi = int(y_v.item())
            else:
                yi = int(g.y.view(-1)[0].item())
            new = PyGData(x=g.x, edge_index=g.edge_index,
                          y=torch.tensor([yi], dtype=torch.long))
            if getattr(g, 'edge_attr', None) is not None:
                new.edge_attr = g.edge_attr
            if getattr(g, 'atom_type_oh', None) is not None:
                new.atom_type_oh = g.atom_type_oh
            test_graphs.append(new)
            test_orig_idx.append(orig)

        # For multitask models output [B, num_tasks]; wrap forward to slice
        # to chosen task so saliency target is single scalar.
        if is_multitask:
            base_model = model
            class _SingleTaskWrap(nn.Module):
                def __init__(self, m, ti):
                    super().__init__()
                    self.m = m
                    self.ti = ti
                def forward(self, data):
                    out = self.m(data)
                    if out.dim() == 1:
                        out = out.unsqueeze(-1)
                    return out[:, self.ti:self.ti+1]
            interp_model = _SingleTaskWrap(base_model, saliency_task_idx)
        else:
            interp_model = model

        ms = None
        if motif_masks is not None:
            ms = {i: motif_masks[test_orig_idx[i]]
                  for i in range(len(test_orig_idx))
                  if test_orig_idx[i] in motif_masks}

        agg, records = run_graph_fidelity(
            model=interp_model,
            test_graphs=test_graphs,
            device=device,
            topk_fracs=topk_fracs,
            seed=seed,
            target_label=1,
            motif_masks=ms,
        )
        agg['test_auc'] = test_auc
        agg['val_auc'] = best_val
        all_aggs.append(agg)
        print(format_raw_log(method_tag, dataset, split_tag, seed, agg))

    # final aggregate over seeds
    if all_aggs:
        agg_keys = [k for k in all_aggs[0].keys() if isinstance(all_aggs[0][k], (int, float))]
        summary = {}
        for k in agg_keys:
            vs = [a[k] for a in all_aggs
                  if k in a and a[k] is not None
                  and not (isinstance(a[k], float) and np.isnan(a[k]))]
            if vs:
                summary[f'{k}_mean'] = float(np.mean(vs))
                summary[f'{k}_std'] = float(np.std(vs))
        parts = [f"method={method_tag}", f"dataset={dataset}",
                 f"split={split_tag}", f"n_seeds={len(all_aggs)}"]
        for k in sorted(summary.keys()):
            parts.append(f"{k}={summary[k]:.4f}")
        print("[GINTERP_V2_AGG] " + " ".join(parts))
