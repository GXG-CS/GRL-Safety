"""GNN baseline (GCN/GAT/SAGE) from-scratch FT + feature-noise corruption eval.

Mirrors BGRL's run_feature_noise.py protocol, but the encoder is initialized
from scratch (no pretrain ckpt). Pick encoder with `--model {gcn,gat,sage}`.

Uniform hyperparameters across all three models and all datasets:
  lr=1e-3, dropout=0.2, weight_decay=1e-4, max_epochs=500, patience=200,
  hidden=768, num_layers=2
"""

import copy
import os
import os.path as osp
import sys
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.supervised.gnn_baseline import GNNEncoderWrapper, METHOD_NAMES
from sgb.data.tag_registry import load as load_tag
from sklearn.metrics import f1_score, roc_auc_score

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG dataset name.')
flags.DEFINE_string('model', 'gcn', 'GNN baseline: gcn, gat, or sage.')
flags.DEFINE_integer('hidden', 768, 'Hidden dim (matches SBERT 768d).')
flags.DEFINE_integer('num_layers', 2, 'Number of encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')

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


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def build_splits(data):
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
        N = data.num_nodes
        tm = _idx_to_mask(s['train'], N)
        vm = _idx_to_mask(s.get('valid', s.get('val')), N)
        tsm = _idx_to_mask(s['test'], N)
        for _ in range(5):
            splits.append({'train': tm, 'val': vm, 'test': tsm})
    else:
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
    return splits


def train_ft(model, data, y, train_mask, val_mask, test_mask, device):
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    best_val, best_state, no_improve = -1.0, None, 0

    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(-1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()

        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FLAGS.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(-1)
        y_true = y[test_mask].cpu().numpy()
        y_pred = pred[test_mask].cpu().numpy()
        test_acc = (pred[test_mask] == y[test_mask]).float().mean().item() * 100.0
        macro_f1 = f1_score(y_true, y_pred, average='macro') * 100.0
    return test_acc, macro_f1


LINK_DATASETS = {"WN18RR", "FB15K237", "goodreads", "ml1m", "ml1m_cls", "protein_hs"}
GRAPH_DATASETS = {"chemhiv", "chempcba", "bace", "bbbp", "cyp450", "muv", "tox21", "toxcast"}


def _build_encoder(in_channels):
    return GNNEncoderWrapper(
        model_name=FLAGS.model,
        in_channels=in_channels,
        hidden_channels=FLAGS.hidden,
        num_layers=FLAGS.num_layers,
        dropout=FLAGS.dropout,
    )


def method_tag():
    return METHOD_NAMES[FLAGS.model.lower()]


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tag = method_tag()
    print(f"[{tag} FT-FN] Using {device}")

    if FLAGS.dataset in LINK_DATASETS:
        _run_link_ft_fn(device)
        return

    if FLAGS.dataset in GRAPH_DATASETS:
        _run_graph_ft_fn(device)
        return

    # ====== Node classification ======
    data, _ = load_tag(FLAGS.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat
    if data.y.dim() > 1:
        data.y = data.y.squeeze()

    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    input_size = data.x.size(1)
    x_clean = data.x.clone()

    print(f"[{tag} FT-FN] {FLAGS.dataset}, N={data.num_nodes}, d={input_size}, C={num_classes}")

    splits = build_splits(data)
    all_results = []

    for split_idx, split in enumerate(splits):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_mask = split['train'].to(device)
        val_mask = split['val'].to(device)
        test_mask = split['test'].to(device)

        data.x = x_clean
        encoder = _build_encoder(input_size)
        model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)

        clean_acc, clean_f1 = train_ft(model, data, y, train_mask, val_mask, test_mask, device)
        all_results.append({"split_idx": split_idx, "sev": 0, "acc": clean_acc, "f1": clean_f1})
        print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
              f"test_acc={clean_acc:.4f} macro_f1={clean_f1:.4f}")

        # Corruption eval on frozen best-val model
        model.eval()
        for sev_idx, sigma_rel in SEVERITIES:
            data.x = apply_feature_noise(x_clean, train_mask, sigma_rel,
                                         noise_seed=split_idx * 100 + sev_idx)
            with torch.no_grad():
                pred = model(data).argmax(-1)
                y_true = y[test_mask].cpu().numpy()
                y_pred = pred[test_mask].cpu().numpy()
                noise_acc = (pred[test_mask] == y[test_mask]).float().mean().item() * 100.0
                noise_f1 = f1_score(y_true, y_pred, average='macro') * 100.0
            all_results.append({"split_idx": split_idx, "sev": sev_idx, "acc": noise_acc, "f1": noise_f1})
            print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                  f"sigma_rel={sigma_rel} test_acc={noise_acc:.4f} macro_f1={noise_f1:.4f}")

        data.x = x_clean

    _aggregate_and_print(tag, all_results, metric_name="acc")


def _aggregate_and_print(tag, all_results, metric_name="acc"):
    print(f"\n=== {tag} FT Feature Noise Results ===")
    grouped_acc = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped_acc[r["sev"]].append(r[metric_name] if metric_name in r else r.get("auc", r["acc"]))
        grouped_f1[r["sev"]].append(r["f1"])
    agg_acc, agg_f1 = {}, {}
    for sev in sorted(grouped_acc.keys()):
        accs = np.array(grouped_acc[sev])
        f1s = np.array(grouped_f1[sev])
        agg_acc[sev] = f"{accs.mean():.2f} ± {accs.std():.2f}"
        agg_f1[sev] = f"{f1s.mean():.2f} ± {f1s.std():.2f}"
        label = "clean" if sev == 0 else f"sev{sev}"
        print(f"  {label:<10} {metric_name}={agg_acc[sev]}  f1={agg_f1[sev]}")
    print(f"[FN_AGG] method={tag} dataset={FLAGS.dataset} "
          f"clean=\"{agg_acc.get(0,'')}\" "
          f"sev1=\"{agg_acc.get(1,'')}\" sev2=\"{agg_acc.get(2,'')}\" "
          f"sev3=\"{agg_acc.get(3,'')}\" sev4=\"{agg_acc.get(4,'')}\" "
          f"sev5=\"{agg_acc.get(5,'')}\" "
          f"clean_f1=\"{agg_f1.get(0,'')}\" "
          f"sev1_f1=\"{agg_f1.get(1,'')}\" sev2_f1=\"{agg_f1.get(2,'')}\" "
          f"sev3_f1=\"{agg_f1.get(3,'')}\" sev4_f1=\"{agg_f1.get(4,'')}\" "
          f"sev5_f1=\"{agg_f1.get(5,'')}\"")


# ======================= Link prediction FT =======================


class FTLinkModel(nn.Module):
    """Encoder (unfrozen) + dot-product link decoder for link prediction FT."""
    def __init__(self, encoder, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)

    def encode(self, data):
        h = self.encoder(data)
        return self.dropout(h)

    def decode(self, z, edge_index):
        return torch.sigmoid((z[edge_index[0]] * z[edge_index[1]]).sum(dim=1))


def _run_link_ft_fn(device):
    from torch_geometric.transforms import RandomLinkSplit, ToUndirected
    from torch_geometric.utils import is_undirected
    from torch_geometric.data import Data

    tag = method_tag()
    data_raw, _ = load_tag(FLAGS.dataset)
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
    print(f"[{tag} FT-FN Link] {FLAGS.dataset}, N={graph.num_nodes}, E={graph.edge_index.size(1)}, d={input_size}")

    all_results = []

    for split_idx in range(5):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        splitter = RandomLinkSplit(num_val=0.1, num_test=0.2,
                                   is_undirected=True,
                                   add_negative_train_samples=True)
        train_data, val_data, test_data = splitter(graph)
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

        encoder = _build_encoder(input_size)
        model = FTLinkModel(encoder, FLAGS.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

        feat_std = train_data.x.std(dim=0, keepdim=True)

        def _eval_split(data_split):
            model.eval()
            with torch.no_grad():
                z = model.encode(data_split)
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
        for epoch in range(1, FLAGS.max_epochs + 1):
            model.train()
            optim.zero_grad()
            z = model.encode(train_data)
            pred = model.decode(z, train_data.edge_label_index)
            loss = F.binary_cross_entropy(pred, train_data.edge_label.float())
            loss.backward()
            optim.step()

            model.eval()
            val_auc, _ = _eval_split(val_data)
            if val_auc > best_val:
                best_val = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= FLAGS.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        clean_auc, clean_f1 = _eval_split(test_data)
        all_results.append({"sev": 0, "auc": clean_auc, "f1": clean_f1})
        print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
              f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
              f"test_auc={clean_auc:.4f} test_f1={clean_f1:.4f}")

        x_clean_tensor = test_data.x.clone()
        for sev_idx, sigma_rel in SEVERITIES:
            g = torch.Generator(device=device).manual_seed(int(split_idx * 100 + sev_idx))
            eps = torch.randn(x_clean_tensor.shape, generator=g, device=device, dtype=x_clean_tensor.dtype)
            test_data.x = x_clean_tensor + sigma_rel * feat_std * eps

            noise_auc, noise_f1 = _eval_split(test_data)
            all_results.append({"sev": sev_idx, "auc": noise_auc, "f1": noise_f1})
            print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                  f"sigma_rel={sigma_rel} test_auc={noise_auc:.4f} test_f1={noise_f1:.4f}")

        test_data.x = x_clean_tensor

    _aggregate_and_print(tag, all_results, metric_name="auc")
    print(f"[METRIC] auc_roc")


# ======================= Graph classification FT =======================


class FTGraphModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        h = self.encoder(data)
        h = global_mean_pool(h, data.batch)
        h = self.dropout(h)
        return self.head(h)


def _run_graph_ft_fn(device):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.data import Data

    tag = method_tag()
    tag_pt = osp.join(_PROJECT_ROOT, "datasets", "TAG", FLAGS.dataset,
                      "processed", "geometric_data_processed.pt")
    merged, slices = torch.load(tag_pt, weights_only=False)
    node_text_feat = merged.node_embs
    n_graphs = slices["y"].shape[0] - 1

    graphs = []
    for i in range(n_graphs):
        ns, ne = slices["x"][i].item(), slices["x"][i+1].item()
        es, ee = slices["edge_index"][i].item(), slices["edge_index"][i+1].item()
        atom_idx = merged.x[ns:ne]
        y_slice = merged.y[slices["y"][i]:slices["y"][i+1]]
        if y_slice.dim() == 1 and y_slice.numel() > 1:
            y_slice = y_slice.unsqueeze(0)
        g = Data(x=node_text_feat[atom_idx],
                 edge_index=merged.edge_index[:, es:ee],
                 y=y_slice)
        graphs.append(g)

    num_tasks = slices["y"][1].item() - slices["y"][0].item()
    is_multitask = num_tasks > 1
    print(f"[{tag} FT-FN] {FLAGS.dataset}: n_graphs={n_graphs}, num_tasks={num_tasks}")

    rng = np.random.RandomState(42)
    perm = rng.permutation(n_graphs)
    n_tr, n_va = int(0.8 * n_graphs), int(0.1 * n_graphs)
    train_idx, val_idx, test_idx = perm[:n_tr], perm[n_tr:n_tr+n_va], perm[n_tr+n_va:]

    input_size = 768

    all_train_feat = torch.cat([graphs[i].x for i in train_idx], dim=0)
    feat_std = all_train_feat.std(dim=0, keepdim=True)

    def _compute_auc(loader, model):
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

    all_results = []
    for split_idx in range(5):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        train_loader = PyGDataLoader([graphs[i] for i in train_idx], batch_size=256, shuffle=True, num_workers=0)
        val_loader = PyGDataLoader([graphs[i] for i in val_idx], batch_size=512, shuffle=False, num_workers=0)
        test_loader = PyGDataLoader([graphs[i] for i in test_idx], batch_size=512, shuffle=False, num_workers=0)

        encoder = _build_encoder(input_size)
        model = FTGraphModel(encoder, num_tasks, FLAGS.dropout).to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

        best_val, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, FLAGS.max_epochs + 1):
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
            val_auc, _ = _compute_auc(val_loader, model)
            if val_auc > best_val:
                best_val = val_auc
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= FLAGS.patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        clean_auc, clean_f1 = _compute_auc(test_loader, model)
        all_results.append({"sev": 0, "auc": clean_auc, "f1": clean_f1})
        print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
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
            noisy_loader = PyGDataLoader(noisy_test, batch_size=512, shuffle=False, num_workers=0)
            noise_auc, noise_f1 = _compute_auc(noisy_loader, model)
            all_results.append({"sev": sev_idx, "auc": noise_auc, "f1": noise_f1})
            print(f"[FN_RAW] method={tag} dataset={FLAGS.dataset} "
                  f"split_idx={split_idx} seed={split_idx} sev={sev_idx} "
                  f"sigma_rel={sigma_rel} test_auc={noise_auc:.4f} test_f1={noise_f1:.4f}")

    _aggregate_and_print(tag, all_results, metric_name="auc")
    print(f"[METRIC] auc_roc")


if __name__ == "__main__":
    app.run(main)
