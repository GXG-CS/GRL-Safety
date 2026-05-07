"""OOD split builders for v2: time shift (node) and size shift (graph).

Follows GOOD (Gui et al. NeurIPS'22) covariate-shift protocol: 60/10/10/20.
  - train_pool = 60% (domain 1, "ID")
  - id_val    = 10% random hold-out from train_pool
  - id_test   = 10% random hold-out from train_pool
  - ood_val   =  5% (domain 2 boundary)
  - ood_test  = 20% (domain 2 tail)

For time shift the "domain" is publication year (sorted ascending).
For size shift the "domain" is graph node count (sorted ascending).
"""

import numpy as np
import torch


def build_time_shift_split(dataset_name, year_tensor, labels, split_seed,
                           train_max_year=None, ood_min_year=None):
    """Covariate-shift split along publication year.

    Two modes:
      * Position-based (default): oldest 60% -> train, newest 35% -> ood_test
      * Year-cutoff: train <= train_max_year, ood_test >= ood_min_year
        (years in between become ood_val; used for a more aggressive temporal gap)

    Args:
        dataset_name: str for logging
        year_tensor: int tensor [N] or [N, 1] of per-node year
        labels: int tensor [N] (-1 for unlabeled)
        split_seed: int, seed for id_val / id_test permutation
        train_max_year: int, if set use year-cutoff mode
        ood_min_year: int, if set use year-cutoff mode
    """
    if year_tensor.dim() == 2:
        year_tensor = year_tensor.squeeze(-1)
    year_cpu = year_tensor.detach().cpu().long()
    labels_cpu = labels.detach().cpu().long()
    num_nodes = int(labels_cpu.numel())

    if labels_cpu.dtype.is_floating_point:
        labeled_bool = ~torch.isnan(labels_cpu)
    else:
        labeled_bool = labels_cpu >= 0
    labeled_idx_all = torch.arange(num_nodes)[labeled_bool]
    labeled_y_all = labels_cpu[labeled_idx_all]
    labeled_year = year_cpu[labeled_idx_all]

    num_classes_total = int(torch.unique(labeled_y_all).numel())

    # Sort labeled nodes by year ascending (oldest first → train pool)
    sort_key = labeled_year * (num_nodes + 1) + labeled_idx_all.long()
    order = torch.argsort(sort_key, descending=False)
    sorted_idx = labeled_idx_all[order]
    sorted_y = labeled_y_all[order]
    sorted_year = labeled_year[order]

    n_labeled = int(sorted_idx.numel())

    if train_max_year is not None and ood_min_year is not None:
        # Year-cutoff mode: aggressive temporal gap
        train_mask = sorted_year <= train_max_year
        ood_test_mask = sorted_year >= ood_min_year
        ood_val_mask = (~train_mask) & (~ood_test_mask)
        train_pool_idx = sorted_idx[train_mask]
        train_pool_y = sorted_y[train_mask]
        ood_val_idx = sorted_idx[ood_val_mask]
        ood_test_idx = sorted_idx[ood_test_mask]
        strategy_str = f'year_cutoff_train<={train_max_year}_ood>={ood_min_year}'
    else:
        # Position-based mode: oldest 60%, next 5% val, newest 35% test
        train_end = int(round(n_labeled * 0.60))
        ood_val_end = int(round(n_labeled * 0.65))
        train_pool_idx = sorted_idx[:train_end]
        train_pool_y = sorted_y[:train_end]
        ood_val_idx = sorted_idx[train_end:ood_val_end]
        ood_test_idx = sorted_idx[ood_val_end:]
        strategy_str = 'good_60_5_35_ascending_year'

    if (train_pool_idx.numel() == 0 or ood_val_idx.numel() == 0
            or ood_test_idx.numel() == 0):
        return _empty_split(dataset_name, split_seed, "time",
                            "empty_bucket", num_classes_total, num_nodes, n_labeled)

    # Carve 15% / 15% of TRAIN POOL as id_val / id_test (fraction of pool, not total)
    n_pool = int(train_pool_idx.numel())
    num_id = max(1, int(round(n_pool * 0.15)))
    if 2 * num_id >= n_pool:
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(rng.permutation(n_pool), dtype=torch.long)
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id: -num_id]
        id_test_idx = shuffled[-num_id:]

    def _range_tuple(v):
        if v.numel() == 0: return (None, None)
        return (int(v.min().item()), int(v.max().item()))

    train_pool_counts = torch.bincount(train_pool_y, minlength=num_classes_total)
    present = int((train_pool_counts > 0).sum().item())
    smallest = int(train_pool_counts[train_pool_counts > 0].min().item()) if present > 0 else 0
    missing_classes = int(num_classes_total - present)

    return {
        "train": actual_train_idx,
        "id_val": id_val_idx,
        "id_test": id_test_idx,
        "ood_val": ood_val_idx,
        "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "shift": "time",
            "strategy": strategy_str,
            "time_shift": "ok",
            "num_classes": num_classes_total,
            "num_nodes_total": num_nodes,
            "n_labeled": n_labeled,
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()),
            "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()),
            "ood_test_size": int(ood_test_idx.numel()),
            "train_pool_year_range": _range_tuple(year_cpu[train_pool_idx]),
            "ood_val_year_range": _range_tuple(year_cpu[ood_val_idx]),
            "ood_test_year_range": _range_tuple(year_cpu[ood_test_idx]),
            "smallest_train_pool_class": smallest,
            "missing_classes_in_train_pool": missing_classes,
        },
    }


def build_size_shift_split(dataset_name, graph_sizes, labels, split_seed):
    """Covariate-shift split along graph-node count (graph-level OOD).

    Args:
        dataset_name: str
        graph_sizes: int tensor [num_graphs], node count per graph
        labels: tensor [num_graphs] or [num_graphs, num_tasks]
        split_seed: int

    Returns:
        dict with keys: train, id_val, id_test, ood_val, ood_test, meta
    """
    sizes_cpu = graph_sizes.detach().cpu().long()
    num_graphs = int(sizes_cpu.numel())
    all_idx = torch.arange(num_graphs)

    # Sort ascending (smallest = train)
    sort_key = sizes_cpu
    order = torch.argsort(sort_key, descending=False)
    sorted_idx = all_idx[order]
    sorted_size = sizes_cpu[order]

    train_end = int(round(num_graphs * 0.60))
    ood_val_end = int(round(num_graphs * 0.65))

    train_pool_idx = sorted_idx[:train_end]
    ood_val_idx = sorted_idx[train_end:ood_val_end]
    ood_test_idx = sorted_idx[ood_val_end:]

    num_id = int(round(num_graphs * 0.10))
    if 2 * num_id >= train_pool_idx.numel():
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(rng.permutation(int(train_pool_idx.numel())),
                               dtype=torch.long)
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id: -num_id]
        id_test_idx = shuffled[-num_id:]

    def _range(v):
        if v.numel() == 0: return (None, None)
        return (int(v.min().item()), int(v.max().item()))

    return {
        "train": actual_train_idx,
        "id_val": id_val_idx,
        "id_test": id_test_idx,
        "ood_val": ood_val_idx,
        "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "shift": "size",
            "strategy": "good_60_5_35_ascending_size",
            "size_shift": "ok",
            "num_graphs_total": num_graphs,
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()),
            "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()),
            "ood_test_size": int(ood_test_idx.numel()),
            "train_pool_size_range": _range(sorted_size[:train_end]),
            "ood_val_size_range": _range(sorted_size[train_end:ood_val_end]),
            "ood_test_size_range": _range(sorted_size[ood_val_end:]),
        },
    }


def _empty_split(dataset_name, split_seed, shift, reason, num_classes, num_nodes, n_labeled):
    return {
        "train": torch.empty(0, dtype=torch.long),
        "id_val": torch.empty(0, dtype=torch.long),
        "id_test": torch.empty(0, dtype=torch.long),
        "ood_val": torch.empty(0, dtype=torch.long),
        "ood_test": torch.empty(0, dtype=torch.long),
        "meta": {"dataset": dataset_name, "split_seed": split_seed, "shift": shift,
                 f"{shift}_shift": "not_applicable", "reason": reason,
                 "num_classes": num_classes, "num_nodes_total": num_nodes,
                 "n_labeled": n_labeled},
    }


def build_homophily_split(dataset_name, edge_index, labels, split_seed):
    """Covariate shift along local homophily h(v) = fraction of neighbors with same label.

    Train on HIGH homophily 60%, OOD test on LOW homophily 35% (reversed from Platonov — hard direction). Per Platonov 2023 /
    EERM, this exposes methods that rely on smoothing assumption.

    Protocol:
      1. For each labeled node v, compute h(v) = #same-label-neighbors / #neighbors.
         Nodes with no labeled neighbors use h=0 (treated as lowest).
      2. Sort ascending.
      3. Train pool = bottom 60% (low homophily / hard).
      4. OOD val = 5% middle, OOD test = top 35% (high homophily / easy).
      5. Carve id_val/id_test (10% each) from train pool via seed-controlled permutation.
    """
    labels_cpu = labels.detach().cpu().long()
    num_nodes = int(labels_cpu.numel())

    if labels_cpu.dtype.is_floating_point:
        labeled_bool = ~torch.isnan(labels_cpu)
    else:
        labeled_bool = labels_cpu >= 0
    labeled_idx_all = torch.arange(num_nodes)[labeled_bool]
    labeled_y_all = labels_cpu[labeled_idx_all]
    num_classes_total = int(torch.unique(labeled_y_all).numel())

    # Compute per-node homophily
    ei = edge_index.detach().cpu().long()
    src, dst = ei[0], ei[1]
    src_label = labels_cpu[src]
    dst_label = labels_cpu[dst]
    same_label = (src_label == dst_label) & (src_label >= 0) & (dst_label >= 0)

    same_count = torch.zeros(num_nodes, dtype=torch.long)
    total_count = torch.zeros(num_nodes, dtype=torch.long)
    same_count.scatter_add_(0, src, same_label.long())
    total_count.scatter_add_(0, src, ((src_label >= 0) & (dst_label >= 0)).long())
    # Also treat undirected (aggregate to both endpoints)
    same_count.scatter_add_(0, dst, same_label.long())
    total_count.scatter_add_(0, dst, ((src_label >= 0) & (dst_label >= 0)).long())

    homo = torch.zeros(num_nodes, dtype=torch.float)
    mask = total_count > 0
    homo[mask] = same_count[mask].float() / total_count[mask].float()
    # nodes with no labeled neighbors: stay 0 (lowest homophily bucket)

    labeled_homo = homo[labeled_idx_all]

    sort_key = labeled_homo * 1000000 + labeled_idx_all.float()  # stable ordering
    order = torch.argsort(sort_key, descending=True)
    sorted_idx = labeled_idx_all[order]
    sorted_y = labeled_y_all[order]
    sorted_homo = labeled_homo[order]

    n_labeled = int(sorted_idx.numel())
    train_end = int(round(n_labeled * 0.60))
    ood_val_end = int(round(n_labeled * 0.65))
    train_pool_idx = sorted_idx[:train_end]
    train_pool_y = sorted_y[:train_end]
    ood_val_idx = sorted_idx[train_end:ood_val_end]
    ood_test_idx = sorted_idx[ood_val_end:]

    if train_pool_idx.numel() == 0 or ood_test_idx.numel() == 0:
        return _empty_split(dataset_name, split_seed, "homophily",
                            "empty_bucket", num_classes_total, num_nodes, n_labeled)

    num_id = int(round(n_labeled * 0.10))
    if 2 * num_id >= train_pool_idx.numel():
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(rng.permutation(int(train_pool_idx.numel())),
                               dtype=torch.long)
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id: -num_id]
        id_test_idx = shuffled[-num_id:]

    def _range_tuple(v):
        if v.numel() == 0: return (None, None)
        return (float(v.min().item()), float(v.max().item()))

    train_pool_counts = torch.bincount(train_pool_y, minlength=num_classes_total)
    present = int((train_pool_counts > 0).sum().item())
    smallest = int(train_pool_counts[train_pool_counts > 0].min().item()) if present > 0 else 0
    missing_classes = int(num_classes_total - present)

    return {
        "train": actual_train_idx,
        "id_val": id_val_idx,
        "id_test": id_test_idx,
        "ood_val": ood_val_idx,
        "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "shift": "homophily",
            "strategy": "top60_bottom35_descending_homophily",
            "homophily_shift": "ok",
            "num_classes": num_classes_total,
            "num_nodes_total": num_nodes,
            "n_labeled": n_labeled,
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()),
            "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()),
            "ood_test_size": int(ood_test_idx.numel()),
            "train_pool_homo_range": _range_tuple(sorted_homo[:train_end]),
            "ood_val_homo_range": _range_tuple(sorted_homo[train_end:ood_val_end]),
            "ood_test_homo_range": _range_tuple(sorted_homo[ood_val_end:]),
            "smallest_train_pool_class": smallest,
            "missing_classes_in_train_pool": missing_classes,
        },
    }


def build_feature_cluster_split(dataset_name, features, labels, split_seed,
                                n_clusters=8, n_ood_clusters=2):
    """Covariate shift via k-means clustering on node features (e.g. SBERT 768d).

    Protocol:
      1. KMeans(k=n_clusters) on features.
      2. Rank clusters by size descending.
      3. Pick smallest `n_ood_clusters` clusters -> ood_test.
      4. From remaining clusters: 90% train pool / 5% ood_val / 5% unused
         then carve 10/10 id_val/id_test from train.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise RuntimeError("sklearn not available; cannot do feature-cluster split")

    labels_cpu = labels.detach().cpu().long()
    num_nodes = int(labels_cpu.numel())
    features_cpu = features.detach().cpu().numpy()

    if labels_cpu.dtype.is_floating_point:
        labeled_bool = ~torch.isnan(labels_cpu)
    else:
        labeled_bool = labels_cpu >= 0
    labeled_idx_all = torch.arange(num_nodes)[labeled_bool]
    num_classes_total = int(torch.unique(labels_cpu[labeled_idx_all]).numel())

    km = KMeans(n_clusters=n_clusters, random_state=split_seed, n_init=5).fit(features_cpu)
    cluster = torch.as_tensor(km.labels_, dtype=torch.long)

    # Cluster sizes over labeled nodes
    labeled_cluster = cluster[labeled_idx_all]
    sizes = torch.bincount(labeled_cluster, minlength=n_clusters)
    # Sort by size ascending -> smallest are OOD
    _, order_clu = torch.sort(sizes, descending=True)
    ood_clusters = set(order_clu[:n_ood_clusters].tolist())
    val_clusters = set(order_clu[n_ood_clusters:n_ood_clusters + 1].tolist())
    train_clusters = set(order_clu[n_ood_clusters + 1:].tolist())

    train_mask = torch.tensor([int(c.item()) in train_clusters for c in labeled_cluster])
    valb_mask = torch.tensor([int(c.item()) in val_clusters for c in labeled_cluster])
    oodt_mask = torch.tensor([int(c.item()) in ood_clusters for c in labeled_cluster])

    train_pool_idx = labeled_idx_all[train_mask]
    ood_val_idx = labeled_idx_all[valb_mask]
    ood_test_idx = labeled_idx_all[oodt_mask]

    if train_pool_idx.numel() == 0 or ood_test_idx.numel() == 0:
        return _empty_split(dataset_name, split_seed, "feature_cluster",
                            "empty_bucket", num_classes_total, num_nodes,
                            int(labeled_idx_all.numel()))

    num_id = int(round(labeled_idx_all.numel() * 0.10))
    if 2 * num_id >= train_pool_idx.numel():
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(rng.permutation(int(train_pool_idx.numel())),
                               dtype=torch.long)
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id: -num_id]
        id_test_idx = shuffled[-num_id:]

    return {
        "train": actual_train_idx,
        "id_val": id_val_idx,
        "id_test": id_test_idx,
        "ood_val": ood_val_idx,
        "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name,
            "split_seed": split_seed,
            "shift": "feature_cluster",
            "strategy": f"kmeans_k{n_clusters}_ood{n_ood_clusters}",
            "feature_cluster_shift": "ok",
            "num_classes": num_classes_total,
            "num_nodes_total": num_nodes,
            "n_labeled": int(labeled_idx_all.numel()),
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()),
            "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()),
            "ood_test_size": int(ood_test_idx.numel()),
            "ood_clusters": sorted(list(ood_clusters)),
            "val_clusters": sorted(list(val_clusters)),
            "cluster_sizes": sizes.tolist(),
        },
    }
