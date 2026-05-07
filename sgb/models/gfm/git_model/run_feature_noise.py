"""
GIT fine-tuning + feature-noise corruption evaluation.

Mirrors the original `finetune.py` flow exactly, then after each split's
training restores the best-val checkpoint and runs a clean eval + 5
feature-noise evals (σ_rel ∈ {0.1, 0.25, 0.5, 1.0, 2.0}).

Results are printed as structured `[FN_RAW]` and `[FN_AGG]` lines so a
downstream aggregator can grep them out of slurm logs.
"""

import os
import os.path as osp
import sys
import collections
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW

from data.finetune_data import get_data
from data.pretrain_data import domain2task, dataset2domain
from model.encoder import Encoder
from model.finetune_model import TaskModel
from utils.utils import seed_everything, load_params, mask2idx, get_scheduler, get_device_from_model, check_path, get_n_params
from utils.args import get_args_finetune
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.split import get_split
from utils.loader import get_ft_loader

from task.node import ft_node, eval_node, eval_node_few_shot
from task.edge import ft_edge, eval_edge, eval_edge_few_show
from task.link_pred import ft_link_pred, eval_link_pred
from task.graph import ft_graph, eval_graph

import wandb
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Feature noise config (per spec)
# -----------------------------------------------------------------------------

SEVERITIES = [
    (1, 0.1),
    (2, 0.25),
    (3, 0.5),
    (4, 1.0),
    (5, 2.0),
]


def apply_feature_noise(
    x: torch.Tensor,
    train_mask: torch.Tensor,
    sigma_rel: float,
    noise_seed: int = 0,
) -> torch.Tensor:
    """Additive relative Gaussian noise, scaled by per-dim std of training nodes."""
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    if train_mask.device != x.device:
        train_mask = train_mask.to(x.device)
    std = x[train_mask].std(dim=0, keepdim=True)
    g = torch.Generator(device=x.device).manual_seed(int(noise_seed))
    eps = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


def get_ft(params):
    task = params["task"]
    if task == "node":
        return ft_node
    elif task == "edge":
        return ft_edge
    elif task == "link_pred":
        return ft_link_pred
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Does not support the task in finetuning.")


def get_eval(params):
    setting = params["setting"]
    task = params["task"]
    if task == "node":
        if setting in ['base', 'base_zero_shot']:
            return eval_node
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_node_few_shot
    elif task == "edge":
        if setting in ['base', 'base_zero_shot']:
            return eval_edge
        elif setting in ['few_shot', 'zero_shot', 'in_context']:
            return eval_edge_few_show
    elif task == "link_pred":
        if setting in ['base']:
            return eval_link_pred
        elif setting in ['base_zero_shot', 'few_shot', 'zero_shot', 'in_context']:
            raise ValueError("Not support the setting yet in evaluation.")
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Does not support the task in evaluation.")


get_loader = get_ft_loader


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Data — load via tag_registry (auto-prepare if not cached)
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag
    graph, _ = load_tag(params["dataset"])
    graph.name = params["dataset"]
    if graph.y is None and hasattr(graph, 'edge_types') and graph.edge_types is not None:
        graph.y = graph.edge_types
    if graph.y.dim() > 1:
        graph.y = graph.y.squeeze()
    graph.num_classes = int(graph.y.max().item()) + 1
    graph.num_nodes = graph.node_text_feat.size(0)
    graph.num_edges = graph.edge_index.size(1)
    print(f"Dataset: {graph.name}, #Nodes: {graph.num_nodes}, #Edges: {graph.num_edges}, #Classes: {graph.num_classes}")

    splits = get_split(graph, params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

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
        print("Loaded pretrained encoder from {}".format(ckpt_dir))
    elif params["pt_data"] != 'na':
        if params['sft_data'] == 'na':
            template = "lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}"
            if params['train_ratio'] != 1.0:
                template += "_{}".format(params['train_ratio'])
            base_path = params['pt_model_path'] if params["sft_data"] == 'na' else params['sft_model_path']
            path = osp.join(base_path,
                            template.format(params['pt_lr'], params['hidden_dim'], params['num_layers'],
                                            params['backbone'], params['pt_feat_p'], params['pt_edge_p'],
                                            params['pt_align_reg_lambda'], params['pt_data']),
                            f"encoder_{params['pt_epochs']}.pt")
        else:
            dir_template = "pt_lr_{}_hidden_{}_layer_{}_backbone_{}_fp_{}_ep_{}_alignreg_{}_pt_data_{}_pt_epochs_{}"
            template = "sft_lr_{}_sft_data_{}"
            path = osp.join(params['sft_model_path'],
                            dir_template.format(params['pt_lr'], params['hidden_dim'], params['num_layers'],
                                                params['backbone'], params['pt_feat_p'], params['pt_feat_p'],
                                                params['pt_align_reg_lambda'], params['pt_data'], params['pt_epochs']),
                            template.format(params['sft_lr'], params['sft_data']),
                            f"encoder_{params['sft_epochs']}.pt")
        check_path(path)
        encoder = load_params(encoder, path)
        print("Load the pretrained model from {}".format(path))

    model = TaskModel(encoder, num_classes=graph.num_classes)
    model = model.to(device)

    logger = Logger()
    all_results = []  # raw per-(split, sev) records
    data_name = params["dataset"]

    for idx, split in enumerate(splits):
        seed_everything(idx)

        if params["bs"] == 0:
            data = deepcopy(graph)
            if params['task'] == 'link_pred':
                data = split(data)
        else:
            data = get_loader(graph, split, params)

        task_model = deepcopy(model)
        optimizer = AdamW(task_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
        stopper = EarlyStopping(patience=params["early_stop"])

        # NEW: track best-val state
        best_val = -float("inf")
        best_state = None

        for epoch in range(1, params["epochs"] + 1):
            loss = finetune(model=task_model, data=data, split=split, optimizer=optimizer, params=params)
            result = evaluate(model=task_model, data=data, split=split, params=params)

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
                    "train/loss_train": loss,
                    "train/train": result['train'],
                    "train/val": result['val'],
                    "train/test": result['test'],
                    "train/metric": result['metric'],
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

        # --- link_pred: data is a tuple (train_data, val_data, test_data) ---
        # NOTE: 'edge' task (KG) uses single-Data else-branch; AUC is enforced via
        # task2metric['edge']='auc' in utils/eval.py so evaluate() returns AUC.
        is_link = params.get("task") == "link_pred"

        if is_link:
            # Each element of the tuple has its own node_text_feat
            original_feats = [d.node_text_feat.clone() for d in data]
            ref_feat = original_feats[0]  # use train split features for std calibration
            train_mask_raw = torch.ones(ref_feat.size(0), dtype=torch.bool, device=ref_feat.device)

            with torch.no_grad():
                clean_result = evaluate(model=task_model, data=data, split=split, params=params)
            clean_auc = float(clean_result.get("test_auc", clean_result["test"]))
            clean_f1 = float(clean_result.get("test_f1", 0.0))
            all_results.append({"split_idx": idx, "seed": idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_auc, "macro_f1": clean_f1})
            print(
                f"[FN_RAW] method=GIT dataset={data_name} "
                f"split_idx={idx} seed={idx} sev=0 sigma_rel=0.0 test_auc={clean_auc:.4f} test_f1={clean_f1:.4f}"
            )

            for sev_idx, sigma_rel in SEVERITIES:
                for i, d in enumerate(data):
                    d.node_text_feat = apply_feature_noise(
                        original_feats[i], train_mask_raw, sigma_rel, noise_seed=idx * 100 + sev_idx
                    )
                with torch.no_grad():
                    noisy_result = evaluate(model=task_model, data=data, split=split, params=params)
                noisy_auc = float(noisy_result.get("test_auc", noisy_result["test"]))
                noisy_f1 = float(noisy_result.get("test_f1", 0.0))
                all_results.append({
                    "split_idx": idx, "seed": idx, "sev": sev_idx,
                    "sigma_rel": sigma_rel, "test_acc": noisy_auc, "macro_f1": noisy_f1,
                })
                print(
                    f"[FN_RAW] method=GIT dataset={data_name} "
                    f"split_idx={idx} seed={idx} sev={sev_idx} sigma_rel={sigma_rel} "
                    f"test_auc={noisy_auc:.4f} test_f1={noisy_f1:.4f}"
                )
                for i, d in enumerate(data):
                    d.node_text_feat = original_feats[i]  # restore

            for i, d in enumerate(data):
                d.node_text_feat = original_feats[i]  # double safety
        else:
            # --- node / edge tasks: data is a single Data object ---
            original_feat = data.node_text_feat.clone() if hasattr(data, "node_text_feat") else None
            if original_feat is None:
                print("[FN_RAW] WARNING: data has no node_text_feat attribute, skipping noise eval")
            else:
                # Get train mask for noise calibration
                # For edge tasks, split["train"] is edge indices, not node mask.
                # Use all nodes for std calibration in that case.
                if params.get("task") == "edge":
                    train_mask_raw = torch.ones(original_feat.size(0), dtype=torch.bool, device=original_feat.device)
                else:
                    train_mask_raw = split.get("train") if isinstance(split, dict) else getattr(split, "train_mask", None)
                    if train_mask_raw is None:
                        train_mask_raw = data.train_mask
                    if not isinstance(train_mask_raw, torch.Tensor):
                        train_mask_raw = torch.tensor(train_mask_raw)
                    train_mask_raw = train_mask_raw.to(original_feat.device)

                # Clean eval (matches reported best-val checkpoint)
                with torch.no_grad():
                    clean_result = evaluate(model=task_model, data=data, split=split, params=params)
                clean_acc = float(clean_result["test"])
                clean_f1 = float(clean_result.get("test_f1", 0.0))
                all_results.append({"split_idx": idx, "seed": idx, "sev": 0, "sigma_rel": 0.0, "test_acc": clean_acc, "macro_f1": clean_f1})
                print(
                    f"[FN_RAW] method=GIT dataset={data_name} "
                    f"split_idx={idx} seed={idx} sev=0 sigma_rel=0.0 test_acc={clean_acc:.4f} macro_f1={clean_f1:.4f}"
                )

                for sev_idx, sigma_rel in SEVERITIES:
                    data.node_text_feat = apply_feature_noise(
                        original_feat, train_mask_raw, sigma_rel, noise_seed=idx * 100 + sev_idx
                    )
                    with torch.no_grad():
                        noisy_result = evaluate(model=task_model, data=data, split=split, params=params)
                    noisy_acc = float(noisy_result["test"])
                    noisy_f1 = float(noisy_result.get("test_f1", 0.0))
                    all_results.append({
                        "split_idx": idx, "seed": idx, "sev": sev_idx,
                        "sigma_rel": sigma_rel, "test_acc": noisy_acc, "macro_f1": noisy_f1,
                    })
                    print(
                        f"[FN_RAW] method=GIT dataset={data_name} "
                        f"split_idx={idx} seed={idx} sev={sev_idx} sigma_rel={sigma_rel} "
                        f"test_acc={noisy_acc:.4f} macro_f1={noisy_f1:.4f}"
                    )
                    data.node_text_feat = original_feat  # restore

                data.node_text_feat = original_feat  # double safety
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

    print(f"\n=== GIT FT Result (clean, best-val, from logger) ===")
    print(f"Train: {best['train']['mean']:.2f} +/- {best['train']['std']:.2f}")
    print(f"Val:   {best['val']['mean']:.2f} +/- {best['val']['std']:.2f}")
    print(f"Test:  {best['test']['mean']:.2f} +/- {best['test']['std']:.2f}")

    # NEW: aggregated feature-noise summary
    if all_results:
        print("\n=== GIT Feature Noise Results (aggregated over splits) ===")
        grouped_acc = collections.defaultdict(list)
        grouped_f1 = collections.defaultdict(list)
        for row in all_results:
            grouped_acc[row["sev"]].append(row["test_acc"])
            grouped_f1[row["sev"]].append(row["macro_f1"])

        label_for_sev = {0: "clean   "}
        for sev_idx, sigma_rel in SEVERITIES:
            label_for_sev[sev_idx] = f"sev{sev_idx} σ={sigma_rel}"

        agg_acc, agg_f1 = {}, {}
        for sev in sorted(grouped_acc.keys()):
            accs = np.array(grouped_acc[sev], dtype=np.float64)
            f1s = np.array(grouped_f1[sev], dtype=np.float64)
            agg_acc[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
            agg_f1[sev] = f"{f1s.mean():.2f} ± {f1s.std():.2f}"
            print(f"  {label_for_sev[sev]:<14}  acc={agg_acc[sev]}  f1={agg_f1[sev]}")

        print(
            f"[FN_AGG] method=GIT dataset={data_name} "
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
            f"sev5_f1=\"{agg_f1.get(5, '')}\""
        )

    wandb.finish()


def main():
    params = get_args_finetune()
    params['data_path'] = osp.join(os.path.dirname(__file__), 'cache_data')
    params['pt_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'pretrain_model')
    params['sft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'sft_model')
    params['ft_model_path'] = osp.join(os.path.dirname(__file__), 'model', 'finetune_model')

    dataset = params["dataset"]
    default_task = domain2task[dataset2domain[dataset]]
    if params['task'] is None:
        params['task'] = default_task
    task = params['task']
    if task == "graph":
        if params['bs'] == 0:
            params['bs'] = 1024

    if params["use_params"]:
        config_path = osp.join(osp.dirname(__file__), "config", f"{params['setting']}.yaml")
        with open(config_path, "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params['base'])
            if task in default_params and dataset in default_params[task]:
                params.update(default_params[task][dataset])

    if params["setting"] in ["zero_shot", "in_context"]:
        params["n_task"] = 500
        params["epochs"] = 1
    elif params['setting'] in ['base_zero_shot']:
        params['epochs'] = 1
        params['repeat'] = 1

    if params['dataset'] == 'products':
        params['bs'] = 1024
    if params['dataset'] == 'chempcba':
        params['n_task'] = 50

    tags = [params['task'], params['setting'], 'feature_noise']
    wandb.init(
        project="GIT-Finetune-FeatureNoise",
        name="Data:{} | SFT:{} | PT-Epoch:{}".format(params["dataset"], params["sft_data"], params["pt_epochs"]),
        config=params,
        mode=params.get("wandb_mode", "offline"),
        tags=tags,
    )
    params = dict(wandb.config)
    print(params)

    if task == "graph":
        _run_graph_fn(params)
    else:
        run(params)


class FTGraphModel_GIT(nn.Module):
    """GIT Encoder (unfrozen) + mean pool + linear head for graph FT."""
    def __init__(self, encoder, hidden_dim, num_tasks, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_tasks)

    def forward(self, batch):
        g_emb = self.encoder.encode_graph(
            batch.node_text_feat, batch.edge_index, batch.batch, pool="mean")
        return self.head(self.dropout(g_emb))


def _run_graph_fn(params):
    """Graph classification FT + feature noise eval — TAG-only, real FT."""
    import collections, copy
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Data
    from sklearn.metrics import roc_auc_score, f1_score

    device = torch.device(f"cuda:{params.get('gpu', 0)}") if torch.cuda.is_available() else torch.device("cpu")
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))

    tag_pt = osp.join(_project_root, "datasets", "TAG", params["dataset"],
                       "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    edge_text_feat = merged.edge_embs
    n_graphs = slices["y"].shape[0] - 1

    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        bond_idx = merged.xe[es:ee]
        y_slice = merged.y[slices["y"][i]:slices["y"][i+1]]
        if y_slice.dim() == 1 and y_slice.numel() > 1:
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

    num_tasks = slices["y"][1].item() - slices["y"][0].item()
    is_multitask = num_tasks > 1
    print(f"[GIT Graph-FN FT] {params['dataset']}: n_graphs={n_graphs}, num_tasks={num_tasks}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr = int(0.8 * n_graphs)
    n_va = int(0.1 * n_graphs)
    train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]

    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    act = params["activation"]
    if isinstance(act, str):
        act = torch.nn.ReLU if act == "relu" else torch.nn.LeakyReLU

    # Compute global feature std from training graphs
    all_train_feat = torch.cat([graphs[i].node_text_feat for i in train_idx], dim=0)
    feat_std = all_train_feat.std(dim=0, keepdim=True)

    hidden_dim = params["hidden_dim"]

    def _compute_auc(loader):
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
            aucs = []
            for t in range(num_tasks):
                mask = ~np.isnan(all_targets[:, t])
                if mask.sum() > 0 and len(np.unique(all_targets[mask, t])) > 1:
                    aucs.append(roc_auc_score(all_targets[mask, t], all_preds[mask, t]))
            auc = np.mean(aucs) * 100.0 if aucs else 50.0
            f1s = []
            for t in range(num_tasks):
                mask = ~np.isnan(all_targets[:, t])
                if mask.sum() > 0 and len(np.unique(all_targets[mask, t])) > 1:
                    pbin = (all_preds[mask, t] > 0.5).astype(int)
                    f1s.append(f1_score(all_targets[mask, t], pbin, zero_division=0))
            f1 = np.mean(f1s) * 100.0 if f1s else 0.0
            return auc, f1
        else:
            probs = all_preds
            pbin = (probs > 0.5).astype(int)
            auc = roc_auc_score(all_targets, probs) * 100.0
            f1 = f1_score(all_targets, pbin, zero_division=0) * 100.0
            return auc, f1

    SEVERITIES = [(1, 0.1), (2, 0.25), (3, 0.5), (4, 1.0), (5, 2.0)]
    all_results = []
    data_name = params["dataset"]
    max_epochs = params.get("max_epochs", 1000)
    patience = params.get("patience", 200)
    lr = params.get("lr", 5e-4)
    wd = params.get("weight_decay", 1e-5)

    for split_idx in range(5):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_loader = PyGDataLoader([graphs[i] for i in train_idx], batch_size=256, shuffle=True, num_workers=0)
        val_loader = PyGDataLoader([graphs[i] for i in val_idx], batch_size=512, shuffle=False, num_workers=0)
        test_loader = PyGDataLoader([graphs[i] for i in test_idx], batch_size=512, shuffle=False, num_workers=0)

        encoder = Encoder(
            input_dim=params["input_dim"], hidden_dim=hidden_dim,
            activation=act, num_layers=params["num_layers"],
            backbone=params["backbone"], normalize=params["normalize"],
            dropout=params["dropout"],
        )
        ckpt_dir = params.get("ckpt_dir")
        if ckpt_dir:
            encoder = load_params(encoder, osp.join(ckpt_dir, "encoder.pt"))
        model = FTGraphModel_GIT(encoder, hidden_dim, num_tasks, dropout=params["dropout"]).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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
            val_auc, _ = _compute_auc(val_loader)
            if val_auc > best_val:
                best_val = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        clean_auc, clean_f1 = _compute_auc(test_loader)
        all_results.append({"sev": 0, "auc": clean_auc, "f1": clean_f1})
        print(f"[FN_RAW] method=GIT dataset={data_name} split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 test_auc={clean_auc:.4f} test_f1={clean_f1:.4f}")

        for sev_idx, sigma_rel in SEVERITIES:
            g_gen = torch.Generator().manual_seed(int(split_idx * 100 + sev_idx))
            noisy_test = []
            for i in test_idx:
                gc = graphs[i].clone()
                eps = torch.randn(gc.node_text_feat.shape, generator=g_gen)
                gc.node_text_feat = gc.node_text_feat + sigma_rel * feat_std * eps
                noisy_test.append(gc)
            noisy_loader = PyGDataLoader(noisy_test, batch_size=512, shuffle=False, num_workers=0)
            noise_auc, noise_f1 = _compute_auc(noisy_loader)
            all_results.append({"sev": sev_idx, "auc": noise_auc, "f1": noise_f1})
            print(f"[FN_RAW] method=GIT dataset={data_name} split_idx={split_idx} seed={split_idx} sev={sev_idx} sigma_rel={sigma_rel} test_auc={noise_auc:.4f} test_f1={noise_f1:.4f}")

    grouped = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped[r["sev"]].append(r["auc"])
        grouped_f1[r["sev"]].append(r["f1"])
    agg = {}
    agg_f1 = {}
    for sev in sorted(grouped.keys()):
        vals = np.array(grouped[sev])
        agg[sev] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        vf = np.array(grouped_f1[sev])
        agg_f1[sev] = f"{vf.mean():.2f} ± {vf.std():.2f}"
    print(f'[FN_AGG] method=GIT dataset={data_name} clean="{agg.get(0,"")}" '
          f'sev1="{agg.get(1,"")}" sev2="{agg.get(2,"")}" sev3="{agg.get(3,"")}" '
          f'sev4="{agg.get(4,"")}" sev5="{agg.get(5,"")}" '
          f'clean_f1="{agg_f1.get(0,"")}" '
          f'sev1_f1="{agg_f1.get(1,"")}" sev2_f1="{agg_f1.get(2,"")}" sev3_f1="{agg_f1.get(3,"")}" '
          f'sev4_f1="{agg_f1.get(4,"")}" sev5_f1="{agg_f1.get(5,"")}"')
    print(f"[METRIC] auc_roc")
    wandb.finish()


if __name__ == "__main__":
    main()
