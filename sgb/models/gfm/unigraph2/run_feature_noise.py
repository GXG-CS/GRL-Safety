"""UniGraph2 FT + feature-noise corruption evaluation.

Loads a joint-pretrained UniGraph2 encoder, attaches a linear head,
fine-tunes end-to-end (encoder unfrozen) on the target dataset, then
evaluates the best-val model under 5 feature-noise severity levels.

Log lines match GraphMAE's convention:
    [FN_RAW]  method=UniGraph2_FT dataset=... split_idx=... seed=...
              sev=... sigma_rel=... test_acc=... macro_f1=...
    [FN_AGG]  method=UniGraph2_FT dataset=... clean="..." sev1="..." ...
"""

import copy
import os
import os.path as osp
import sys
import argparse
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import f1_score, roc_auc_score

_UG2_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_UG2_DIR, "..", "..", ".."))
if _UG2_DIR not in sys.path:
    sys.path.insert(0, _UG2_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.unigraph2 import UniGraph2  # type: ignore
from sgb.data.tag_registry import load as load_tag


SEVERITIES = [
    (1, 0.1),
    (2, 0.25),
    (3, 0.5),
    (4, 1.0),
    (5, 2.0),
]


def apply_feature_noise(x, train_mask, sigma_rel, noise_seed):
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    std = x[train_mask].std(dim=0, keepdim=True)
    g = torch.Generator(device=x.device).manual_seed(int(noise_seed))
    eps = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


def build_model(num_features=768, num_hidden=768, num_layers=3,
                num_experts=8, num_selected_experts=2,
                feat_drop_rate=0.1, edge_mask_rate=0.1,
                gamma=2.0, lambda_spd=0.5):
    return UniGraph2(
        input_dims={"text": num_features},
        hidden_dim=num_hidden,
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        num_layers=num_layers,
        feat_drop_rate=feat_drop_rate,
        edge_mask_rate=edge_mask_rate,
        gamma=gamma,
        lambda_spd=lambda_spd,
    )


class FTModel(nn.Module):
    """UniGraph2 encoder (unfrozen) + linear head for node classification.

    Calls `pre_model.forward(g, {"text": x}, return_embeddings=True)` to get
    node embeddings. Head is a dropout + linear classifier on top.
    """

    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)

    def forward(self, g, x):
        h = self.pre_model(g, {"text": x}, spd_matrix=None, return_embeddings=True)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def load_dataset(name, device):
    data, _ = load_tag(name)

    if data.x is not None and data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat[data.x].float()
    elif data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        feat = data.x.float()
    elif hasattr(data, 'node_text_feat'):
        feat = data.node_text_feat.float()
    else:
        raise RuntimeError(f"Cannot extract 768d features for {name}")

    y = data.y.squeeze() if data.y is not None and data.y.dim() > 1 else data.y

    src, dst = data.edge_index[0], data.edge_index[1]
    g = dgl.graph((src, dst), num_nodes=feat.size(0))
    g = g.remove_self_loop().add_self_loop()

    N = feat.size(0)
    splits = []
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        for i in range(5):
            j = i % avail
            splits.append({
                'train': data.train_masks[j].bool(),
                'val': data.val_masks[j].bool(),
                'test': data.test_masks[j].bool(),
            })
    elif hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
        for _ in range(5):
            splits.append({'train': tm, 'val': vm, 'test': tsm})
    elif hasattr(data, 'train_mask') and data.train_mask is not None:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            avail = tm.size(1)
            for i in range(5):
                j = i % avail
                splits.append({
                    'train': tm[:, j].bool(),
                    'val': vm[:, j].bool(),
                    'test': (tsm[:, j] if tsm.dim() == 2 else tsm).bool(),
                })
        else:
            for _ in range(5):
                splits.append({'train': tm.bool(), 'val': vm.bool(), 'test': tsm.bool()})
    else:
        raise RuntimeError(f"No splits for {name}")

    return g.to(device), feat.to(device), y.long().to(device), splits


def train_ft(model, g, feat, y, train_mask, val_mask, test_mask, device,
             max_epochs=500, patience=100, lr=1e-3, wd=5e-4):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(g, feat)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            pred = model(g, feat).argmax(-1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(g, feat).argmax(-1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item() * 100.0
        macro_f1 = f1_score(y_true, y_pred, average='macro') * 100.0
    return test_acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--max_epochs", default=500, type=int)
    parser.add_argument("--patience", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[UG2 FT-FN] Using {device}")

    if args.dataset in LINK_DATASETS:
        _run_link_ft_fn(device, args)
        return

    if args.dataset in GRAPH_DATASETS:
        _run_graph_ft_fn(device, args)
        return

    g, feat_clean, y, splits = load_dataset(args.dataset, device)
    num_classes = int(y.max().item()) + 1
    print(f"[UG2 FT-FN] {args.dataset}, N={g.num_nodes()}, C={num_classes}")

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    all_results = []

    for split_idx, split in enumerate(splits):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_mask = split['train'].to(device)
        val_mask = split['val'].to(device)
        test_mask = split['test'].to(device)

        pre_model = build_model(num_features=feat_clean.size(1))
        pre_model.load_state_dict(state, strict=False)
        model = FTModel(pre_model, num_hidden=768, num_classes=num_classes,
                        dropout=args.dropout).to(device)

        clean_acc, clean_f1 = train_ft(
            model, g, feat_clean, y, train_mask, val_mask, test_mask,
            device, args.max_epochs, args.patience, args.lr, args.wd,
        )
        all_results.append({"split_idx": split_idx, "sev": 0, "acc": clean_acc, "f1": clean_f1})
        print(
            f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
            f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
            f"test_acc={clean_acc:.4f} macro_f1={clean_f1:.4f}"
        )

        model.eval()
        for sev_idx, sigma_rel in SEVERITIES:
            feat_noisy = apply_feature_noise(
                feat_clean, train_mask, sigma_rel,
                noise_seed=split_idx * 100 + sev_idx,
            )
            with torch.no_grad():
                pred = model(g, feat_noisy).argmax(-1)
                y_true = y[test_mask].cpu().numpy()
                y_pred = pred[test_mask].cpu().numpy()
                noise_acc = (pred[test_mask] == y[test_mask]).float().mean().item() * 100.0
                noise_f1 = f1_score(y_true, y_pred, average='macro') * 100.0
            all_results.append(
                {"split_idx": split_idx, "sev": sev_idx, "acc": noise_acc, "f1": noise_f1}
            )
            print(
                f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
                f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                f"sigma_rel={sigma_rel} test_acc={noise_acc:.4f} macro_f1={noise_f1:.4f}"
            )

    print(f"\n=== UniGraph2 FT Feature Noise Results ===")
    grouped_acc = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped_acc[r["sev"]].append(r["acc"])
        grouped_f1[r["sev"]].append(r["f1"])
    agg_acc, agg_f1 = {}, {}
    for sev in sorted(grouped_acc.keys()):
        accs = np.array(grouped_acc[sev])
        f1s = np.array(grouped_f1[sev])
        agg_acc[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
        agg_f1[sev] = f"{f1s.mean():.2f} ± {f1s.std():.2f}"
        label = "clean" if sev == 0 else f"sev{sev}"
        print(f"  {label:<10} acc={agg_acc[sev]}  f1={agg_f1[sev]}")
    print(
        f"[FN_AGG] method=UniGraph2_FT dataset={args.dataset} "
        f"clean=\"{agg_acc.get(0,'')}\" "
        f"sev1=\"{agg_acc.get(1,'')}\" sev2=\"{agg_acc.get(2,'')}\" "
        f"sev3=\"{agg_acc.get(3,'')}\" sev4=\"{agg_acc.get(4,'')}\" "
        f"sev5=\"{agg_acc.get(5,'')}\" "
        f"clean_f1=\"{agg_f1.get(0,'')}\" "
        f"sev1_f1=\"{agg_f1.get(1,'')}\" sev2_f1=\"{agg_f1.get(2,'')}\" "
        f"sev3_f1=\"{agg_f1.get(3,'')}\" sev4_f1=\"{agg_f1.get(4,'')}\" "
        f"sev5_f1=\"{agg_f1.get(5,'')}\""
    )


GRAPH_DATASETS = {"chemhiv", "chempcba", "bace", "bbbp", "cyp450", "muv", "tox21", "toxcast"}
LINK_DATASETS = {"WN18RR", "FB15K237", "goodreads", "ml1m", "ml1m_cls", "protein_hs"}


# ======================= Link prediction FT =======================


class FTLinkModel_UG2(nn.Module):
    """UniGraph2 encoder (unfrozen) + dot-product link decoder."""

    def __init__(self, pre_model, dropout):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)

    def encode(self, edge_index, x, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g = g.remove_self_loop().add_self_loop().to(x.device)
        h = self.pre_model(g, {"text": x}, spd_matrix=None, return_embeddings=True)
        return self.dropout(h)

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))


def _run_link_ft_fn(device, args):
    from torch_geometric.transforms import RandomLinkSplit, ToUndirected
    from torch_geometric.utils import is_undirected
    from torch_geometric.data import Data

    data_raw, _ = load_tag(args.dataset)
    if data_raw.x is None:
        node_feat = data_raw.node_text_feat
    elif data_raw.x.dtype == torch.long and hasattr(data_raw, 'node_text_feat'):
        node_feat = data_raw.node_text_feat[data_raw.x] if data_raw.x.dim() == 1 else data_raw.node_text_feat
    elif data_raw.x.ndim == 2 and data_raw.x.size(1) != 768 and hasattr(data_raw, 'node_text_feat'):
        node_feat = data_raw.node_text_feat
    else:
        node_feat = data_raw.x if data_raw.x.size(1) == 768 else data_raw.node_text_feat

    graph = Data(x=node_feat, edge_index=data_raw.edge_index)
    if not is_undirected(graph.edge_index):
        graph = ToUndirected()(graph)

    input_size = graph.x.size(1)
    num_nodes = graph.num_nodes
    print(f"[UG2 FT-FN Link] {args.dataset}, N={num_nodes}, E={graph.edge_index.size(1)}")

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    all_results = []
    for split_idx in range(5):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        splitter = RandomLinkSplit(num_val=0.1, num_test=0.2,
                                   is_undirected=True,
                                   add_negative_train_samples=True)
        train_data, val_data, test_data = splitter(graph)
        train_data, val_data, test_data = train_data.to(device), val_data.to(device), test_data.to(device)

        pre_model = build_model(num_features=input_size)
        pre_model.load_state_dict(state, strict=False)
        model = FTLinkModel_UG2(pre_model, args.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        feat_std = train_data.x.std(dim=0, keepdim=True)

        def _eval_split(data_split):
            model.eval()
            with torch.no_grad():
                z = model.encode(data_split.edge_index, data_split.x, data_split.num_nodes)
                pred = model.decode(z, data_split.edge_label_index)
            y_np = data_split.edge_label.cpu().numpy()
            pred_np = pred.cpu().numpy()
            try:
                auc = roc_auc_score(y_np, pred_np) * 100.0
            except ValueError:
                auc = 50.0
            pred_bin = (pred_np >= 0.5).astype(int)
            f1 = f1_score(y_np.astype(int), pred_bin, average='macro', zero_division=0) * 100.0
            return auc, f1

        best_val, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            optim.zero_grad()
            z = model.encode(train_data.edge_index, train_data.x, train_data.num_nodes)
            pred = model.decode(z, train_data.edge_label_index)
            F.binary_cross_entropy(pred, train_data.edge_label.float()).backward()
            optim.step()
            model.eval()
            val_auc, _ = _eval_split(val_data)
            if val_auc > best_val:
                best_val = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    break

        model.load_state_dict(best_state)
        clean_auc, clean_f1 = _eval_split(test_data)
        all_results.append({"sev": 0, "auc": clean_auc, "f1": clean_f1})
        print(f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
              f"test_auc={clean_auc:.4f} test_f1={clean_f1:.4f}")

        x_clean = test_data.x.clone()
        for sev_idx, sigma_rel in SEVERITIES:
            gen = torch.Generator(device=device).manual_seed(int(split_idx * 100 + sev_idx))
            eps = torch.randn(x_clean.shape, generator=gen, device=device, dtype=x_clean.dtype)
            test_data.x = x_clean + sigma_rel * feat_std * eps
            noise_auc, noise_f1 = _eval_split(test_data)
            all_results.append({"sev": sev_idx, "auc": noise_auc, "f1": noise_f1})
            print(f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                  f"sigma_rel={sigma_rel} test_auc={noise_auc:.4f} test_f1={noise_f1:.4f}")
        test_data.x = x_clean

    grouped = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped[r["sev"]].append(r["auc"])
        grouped_f1[r["sev"]].append(r["f1"])
    agg, agg_f1 = {}, {}
    for sev in sorted(grouped.keys()):
        vals = np.array(grouped[sev])
        agg[sev] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        vf = np.array(grouped_f1[sev])
        agg_f1[sev] = f"{vf.mean():.2f} ± {vf.std():.2f}"
    print(f"[FN_AGG] method=UniGraph2_FT dataset={args.dataset} "
          f"clean=\"{agg.get(0,'')}\" "
          f"sev1=\"{agg.get(1,'')}\" sev2=\"{agg.get(2,'')}\" "
          f"sev3=\"{agg.get(3,'')}\" sev4=\"{agg.get(4,'')}\" "
          f"sev5=\"{agg.get(5,'')}\" "
          f"clean_f1=\"{agg_f1.get(0,'')}\" "
          f"sev1_f1=\"{agg_f1.get(1,'')}\" sev2_f1=\"{agg_f1.get(2,'')}\" "
          f"sev3_f1=\"{agg_f1.get(3,'')}\" sev4_f1=\"{agg_f1.get(4,'')}\" "
          f"sev5_f1=\"{agg_f1.get(5,'')}\"")


# ======================= Graph classification FT =======================


class FTGraphModel_UG2(nn.Module):
    """UniGraph2 encoder + global_mean_pool + linear head for graph classification."""

    def __init__(self, pre_model, num_hidden, num_classes, dropout=0.5):
        super().__init__()
        self.pre_model = pre_model
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_hidden, num_classes)

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool
        src, dst = batch.edge_index[0], batch.edge_index[1]
        g = dgl.graph((src, dst), num_nodes=batch.x.size(0))
        g = g.remove_self_loop().add_self_loop().to(batch.x.device)
        h = self.pre_model(g, {"text": batch.x}, spd_matrix=None, return_embeddings=True)
        h = global_mean_pool(h, batch.batch)
        h = self.dropout(h)
        return self.head(h)


def _run_graph_ft_fn(device, args):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Data
    import os.path as osp

    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", args.dataset,
                       "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs if hasattr(merged, 'node_embs') else merged.node_text_feat
    n_graphs = slices["y"].shape[0] - 1

    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        g = Data(x=node_text_feat[atom_idx],
                 edge_index=merged.edge_index[:, es:ee],
                 y=merged.y[slices["y"][i]:slices["y"][i+1]])
        graphs.append(g)

    num_tasks = slices["y"][1].item() - slices["y"][0].item()
    is_multitask = num_tasks > 1
    print(f"[UG2 FT-FN Graph] {args.dataset}: n_graphs={n_graphs}, num_tasks={num_tasks}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr, n_va = int(0.8 * n_graphs), int(0.1 * n_graphs)
    train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    all_train_feat = torch.cat([graphs[i].x for i in train_idx], dim=0)
    feat_std = all_train_feat.std(dim=0, keepdim=True)

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
            aucs, f1s = [], []
            for t in range(num_tasks):
                mask = ~np.isnan(all_targets[:, t])
                if mask.sum() > 0 and len(np.unique(all_targets[mask, t])) > 1:
                    aucs.append(roc_auc_score(all_targets[mask, t], all_preds[mask, t]))
                    pbin = (all_preds[mask, t] > 0.5).astype(int)
                    f1s.append(f1_score(all_targets[mask, t], pbin, zero_division=0))
            return (np.mean(aucs)*100 if aucs else 50.0), (np.mean(f1s)*100 if f1s else 0.0)
        else:
            auc = roc_auc_score(all_targets, all_preds) * 100.0
            pbin = (all_preds > 0.5).astype(int)
            return auc, f1_score(all_targets, pbin, zero_division=0) * 100.0

    all_results = []
    for split_idx in range(5):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_loader = PyGDataLoader([graphs[i] for i in train_idx], batch_size=256, shuffle=True)
        val_loader = PyGDataLoader([graphs[i] for i in val_idx], batch_size=512, shuffle=False)
        test_loader = PyGDataLoader([graphs[i] for i in test_idx], batch_size=512, shuffle=False)

        pre_model = build_model(num_features=768)
        pre_model.load_state_dict(state, strict=False)
        model = FTGraphModel_UG2(pre_model, num_hidden=768, num_classes=num_tasks,
                                 dropout=args.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        best_val, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, args.max_epochs + 1):
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
                if no_improve >= args.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()
        clean_auc, clean_f1 = _compute_auc(test_loader)
        all_results.append({"sev": 0, "auc": clean_auc, "f1": clean_f1})
        print(f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
              f"test_auc={clean_auc:.4f} test_f1={clean_f1:.4f}")

        for sev_idx, sigma_rel in SEVERITIES:
            g_gen = torch.Generator().manual_seed(int(split_idx * 100 + sev_idx))
            noisy_test = []
            for i in test_idx:
                gc = graphs[i].clone()
                eps = torch.randn(gc.x.shape, generator=g_gen)
                gc.x = gc.x + sigma_rel * feat_std * eps
                noisy_test.append(gc)
            noisy_loader = PyGDataLoader(noisy_test, batch_size=512, shuffle=False)
            noise_auc, noise_f1 = _compute_auc(noisy_loader)
            all_results.append({"sev": sev_idx, "auc": noise_auc, "f1": noise_f1})
            print(f"[FN_RAW] method=UniGraph2_FT dataset={args.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                  f"sigma_rel={sigma_rel} test_auc={noise_auc:.4f} test_f1={noise_f1:.4f}")

    grouped = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped[r["sev"]].append(r["auc"])
        grouped_f1[r["sev"]].append(r["f1"])
    agg, agg_f1 = {}, {}
    for sev in sorted(grouped.keys()):
        vals = np.array(grouped[sev])
        agg[sev] = f"{vals.mean():.2f} ± {vals.std():.2f}"
        vf = np.array(grouped_f1[sev])
        agg_f1[sev] = f"{vf.mean():.2f} ± {vf.std():.2f}"
    print(f"[FN_AGG] method=UniGraph2_FT dataset={args.dataset} "
          f"clean=\"{agg.get(0,'')}\" "
          f"sev1=\"{agg.get(1,'')}\" sev2=\"{agg.get(2,'')}\" "
          f"sev3=\"{agg.get(3,'')}\" sev4=\"{agg.get(4,'')}\" "
          f"sev5=\"{agg.get(5,'')}\" "
          f"clean_f1=\"{agg_f1.get(0,'')}\" "
          f"sev1_f1=\"{agg_f1.get(1,'')}\" sev2_f1=\"{agg_f1.get(2,'')}\" "
          f"sev3_f1=\"{agg_f1.get(3,'')}\" sev4_f1=\"{agg_f1.get(4,'')}\" "
          f"sev5_f1=\"{agg_f1.get(5,'')}\"")


if __name__ == "__main__":
    main()
