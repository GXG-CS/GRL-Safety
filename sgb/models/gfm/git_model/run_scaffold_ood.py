"""GIT FT + scaffold-OOD eval on BBBP / BACE.

GIT uses Data with `node_text_feat` field (not `x`). Mirrors run_feature_noise.py
graph path but injects scaffold/random splits via sgb.data.scaffold_split.
"""
import collections
import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score, roc_auc_score

_GIT_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_GIT_DIR, "..", "..", ".."))
if _GIT_DIR not in sys.path:
    sys.path.insert(0, _GIT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.args import get_args_finetune
from model.encoder import Encoder
from utils.utils import load_params
from data.pretrain_data import domain2task, dataset2domain  # noqa: F401
from sgb.models.gfm.git_model.run_feature_noise import FTGraphModel_GIT
from sgb.data.scaffold_split import load_splits


def _build_graphs(dataset_name):
    from torch_geometric.data import Data
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", dataset_name,
                      "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    edge_text_feat = merged.edge_embs
    n_graphs = slices["y"].shape[0] - 1

    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i + 1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i + 1].item()
        atom_idx = merged.x[ns:ne]
        bond_idx = merged.xe[es:ee]
        y_slice = merged.y[slices["y"][i]:slices["y"][i + 1]]
        if y_slice.dim() == 1 and y_slice.numel() > 1:
            y_slice = y_slice.unsqueeze(0)
        graphs.append(Data(
            x=atom_idx,
            edge_index=merged.edge_index[:, es:ee],
            xe=bond_idx,
            y=y_slice,
            node_text_feat=node_text_feat[atom_idx],
            edge_text_feat=edge_text_feat[bond_idx],
        ))
    num_tasks = int(slices["y"][1].item() - slices["y"][0].item())
    return graphs, num_tasks, int(n_graphs)


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


def _train_one(graphs, num_tasks, split, seed, params, device):
    from torch_geometric.loader import DataLoader as PyGDataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = split["train"].tolist()
    val_idx = split["val"].tolist()
    test_idx = split["test"].tolist()

    train_loader = PyGDataLoader([graphs[i] for i in train_idx], batch_size=256, shuffle=True, num_workers=0)
    val_loader = PyGDataLoader([graphs[i] for i in val_idx], batch_size=512, shuffle=False, num_workers=0)
    test_loader = PyGDataLoader([graphs[i] for i in test_idx], batch_size=512, shuffle=False, num_workers=0)

    act = params["activation"]
    if isinstance(act, str):
        act = torch.nn.ReLU if act == "relu" else torch.nn.LeakyReLU

    encoder = Encoder(
        input_dim=params["input_dim"], hidden_dim=params["hidden_dim"],
        activation=act, num_layers=params["num_layers"],
        backbone=params["backbone"], normalize=params["normalize"],
        dropout=params["dropout"],
    )
    ckpt_dir = params.get("ckpt_dir")
    if ckpt_dir:
        encoder = load_params(encoder, osp.join(ckpt_dir, "encoder.pt"))

    model = FTGraphModel_GIT(encoder, params["hidden_dim"], num_tasks,
                             dropout=params["dropout"]).to(device)
    optim = torch.optim.AdamW(model.parameters(),
                              lr=params.get("lr", 5e-4),
                              weight_decay=params.get("weight_decay", 1e-5))

    is_multitask = num_tasks > 1
    max_epochs = params.get("max_epochs", 500)
    patience = params.get("patience", 200)
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
    return _compute_auc(test_loader, model, num_tasks, device)


def main():
    params = get_args_finetune()
    params['data_path'] = osp.join(os.path.dirname(__file__), 'cache_data')
    params['pt_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')
    params['sft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'sft_model')
    params['ft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'finetune_model')
    params['task'] = 'graph'

    if params["use_params"]:
        config_path = osp.join(osp.dirname(__file__), "config", f"{params['setting']}.yaml")
        with open(config_path, "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params['base'])
            if 'graph' in default_params and params['dataset'] in default_params['graph']:
                params.update(default_params['graph'][params['dataset']])

    n_seeds = int(params.get("n_seeds", 5))
    device = torch.device(f"cuda:{params.get('gpu', 0)}") if torch.cuda.is_available() else torch.device("cpu")

    dataset = params["dataset"]
    print(f"[GIT FT-Scaffold] device={device} dataset={dataset}")

    graphs, num_tasks, n_graphs = _build_graphs(dataset)
    print(f"[GIT FT-Scaffold] n_graphs={n_graphs} num_tasks={num_tasks}")

    splits = load_splits(dataset, n_graphs)
    for k in ("random", "scaffold"):
        sp = splits[k]
        print(f"[GIT FT-Scaffold] {k:8s} train={len(sp['train'])} val={len(sp['val'])} test={len(sp['test'])}")

    results = {"random": [], "scaffold": []}
    for split_type in ("random", "scaffold"):
        sp = splits[split_type]
        for seed in range(n_seeds):
            test_auc, test_f1 = _train_one(graphs, num_tasks, sp, seed, params, device)
            results[split_type].append(test_auc)
            print(f"[SCAFFOLD_RAW] method=GIT dataset={dataset} "
                  f"split_type={split_type} seed={seed} "
                  f"test_auc={test_auc:.4f} test_f1={test_f1:.4f}")

    rand_arr = np.asarray(results["random"])
    scaf_arr = np.asarray(results["scaffold"])
    gap = float(rand_arr.mean() - scaf_arr.mean())
    print(f"[SCAFFOLD_AGG] method=GIT dataset={dataset} "
          f"random_auc_mean={rand_arr.mean():.4f} random_auc_std={rand_arr.std():.4f} "
          f"scaffold_auc_mean={scaf_arr.mean():.4f} scaffold_auc_std={scaf_arr.std():.4f} "
          f"gap={gap:.4f} n_seeds={n_seeds}")
    print("[METRIC] auc_roc")


if __name__ == "__main__":
    main()
