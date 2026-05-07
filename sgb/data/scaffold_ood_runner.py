"""Shared scaffold-OOD FT runner.

Each method-specific `run_scaffold_ood.py` provides a model factory and
calls `run_scaffold_ood(...)`. The runner handles:
  - Loading TAG cache + building per-graph PyG Data objects
  - Loading scaffold + random splits via sgb.data.scaffold_split
  - For each (split_type, seed): build a fresh model, FT, eval ROC-AUC
  - Emit [SCAFFOLD_RAW] and [SCAFFOLD_AGG] log lines
"""
from __future__ import annotations

import copy
import os.path as osp
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))


def _build_graphs(dataset_name: str):
    from torch_geometric.data import Data
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", dataset_name,
                      "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    n_graphs = slices["y"].shape[0] - 1

    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i + 1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i + 1].item()
        atom_idx = merged.x[ns:ne]
        y_slice = merged.y[slices["y"][i]:slices["y"][i + 1]]
        if y_slice.dim() == 1 and y_slice.numel() > 1:
            y_slice = y_slice.unsqueeze(0)
        graphs.append(Data(
            x=node_text_feat[atom_idx],
            edge_index=merged.edge_index[:, es:ee],
            y=y_slice,
        ))
    num_tasks = slices["y"][1].item() - slices["y"][0].item()
    return graphs, int(num_tasks), int(n_graphs)


def _compute_auc(loader, model, num_tasks, device):
    is_multitask = num_tasks > 1
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            if is_multitask:
                preds = torch.sigmoid(logits).cpu()
                targets = batch.y.float().view(-1, num_tasks).cpu()
            else:
                preds = torch.sigmoid(logits.squeeze(-1)).cpu()
                targets = batch.y.float().cpu()
            all_preds.append(preds)
            all_targets.append(targets)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    if is_multitask:
        aucs, f1s = [], []
        for t in range(num_tasks):
            mask = ~np.isnan(all_targets[:, t])
            if mask.sum() > 0 and len(np.unique(all_targets[mask, t])) > 1:
                aucs.append(roc_auc_score(all_targets[mask, t], all_preds[mask, t]))
                pbin = (all_preds[mask, t] > 0.5).astype(int)
                f1s.append(f1_score(all_targets[mask, t], pbin, zero_division=0))
        return (float(np.mean(aucs) * 100.0) if aucs else 50.0,
                float(np.mean(f1s) * 100.0) if f1s else 0.0)
    pbin = (all_preds > 0.5).astype(int)
    return (float(roc_auc_score(all_targets, all_preds) * 100.0),
            float(f1_score(all_targets, pbin, zero_division=0) * 100.0))


def run_scaffold_ood(
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
    device: torch.device = None,
):
    """FT a freshly-built model under random + scaffold split, log AUCs."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from sgb.data.scaffold_split import load_splits

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if dataset not in {"bace", "tox21"}:
        raise ValueError(f"scaffold-OOD supports bace/tox21, got {dataset}")

    print(f"[{method_tag} FT-Scaffold] device={device} dataset={dataset}")
    graphs, num_tasks, n_graphs = _build_graphs(dataset)
    is_multitask = num_tasks > 1
    print(f"[{method_tag} FT-Scaffold] n_graphs={n_graphs} num_tasks={num_tasks}")

    splits = load_splits(dataset, n_graphs)
    for k in ("random", "scaffold"):
        sp = splits[k]
        print(f"[{method_tag} FT-Scaffold] {k:8s} train={len(sp['train'])} "
              f"val={len(sp['val'])} test={len(sp['test'])}")

    results = {"random": [], "scaffold": []}
    for split_type in ("random", "scaffold"):
        sp = splits[split_type]
        train_idx = sp["train"].tolist()
        val_idx = sp["val"].tolist()
        test_idx = sp["test"].tolist()

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_loader = PyGDataLoader([graphs[i] for i in train_idx],
                                         batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = PyGDataLoader([graphs[i] for i in val_idx],
                                       batch_size=512, shuffle=False, num_workers=0)
            test_loader = PyGDataLoader([graphs[i] for i in test_idx],
                                        batch_size=512, shuffle=False, num_workers=0)

            model = build_ft_model(768, num_tasks, dropout, device)
            optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
                        loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
                    else:
                        loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), batch.y.float())
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
            results[split_type].append(test_auc)
            print(f"[SCAFFOLD_RAW] method={method_tag} dataset={dataset} "
                  f"split_type={split_type} seed={seed} "
                  f"test_auc={test_auc:.4f} test_f1={test_f1:.4f}")

    rand_arr = np.asarray(results["random"])
    scaf_arr = np.asarray(results["scaffold"])
    gap = float(rand_arr.mean() - scaf_arr.mean())
    print(f"[SCAFFOLD_AGG] method={method_tag} dataset={dataset} "
          f"random_auc_mean={rand_arr.mean():.4f} random_auc_std={rand_arr.std():.4f} "
          f"scaffold_auc_mean={scaf_arr.mean():.4f} scaffold_auc_std={scaf_arr.std():.4f} "
          f"gap={gap:.4f} n_seeds={n_seeds}")
    print(f"[METRIC] auc_roc")
    return results
