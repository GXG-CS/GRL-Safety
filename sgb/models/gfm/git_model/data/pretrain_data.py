import os
import os.path as osp
import math
import torch
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from data.ofa_dataset import MolOFADataset
from torch_geometric.transforms import NormalizeFeatures, ToUndirected, RemoveIsolatedNodes
from torch_geometric.utils import to_undirected

from data.finetune_data import datasets, citation_datasets, ecommerce_datasets, kg_datasets, molecule_datasets, \
    temporal_datasets, get_data

pretrain_datasets = {
    'default': ['arxiv', 'products', 'WN18RR', 'FB15K237', 'chemblpre', 'chempcba'],
    'citation': citation_datasets,
    'ecommerce': ecommerce_datasets,
    'kg': kg_datasets,
    'molecule': molecule_datasets,
    'cora': ['cora'],
    'citeseer': ['citeseer'],
    'pubmed': ['pubmed'],
    'dblp': ['dblp'],
    'arxiv23': ['arxiv23'],
    'arxiv': ['arxiv'],
    'bookhis': ['bookhis'],
    'bookchild': ['bookchild'],
    'elecomp': ['elecomp'],
    'elephoto': ['elephoto'],
    'sportsfit': ['sportsfit'],
    'amazonratings': ['amazonratings'],
    'products': ['products'],
    'chemblpre': ['chemblpre'],
    'chempcba': ['chempcba'],
    'chemhiv': ['chemhiv'],
    'bbbp': ['bbbp'],
    'bace': ['bace'],
    'toxcast': ['toxcast'],
    'cyp450': ['cyp450'],
    'tox21': ['tox21'],
    'muv': ['muv'],
    'WN18RR': ['WN18RR'],
    'FB15K237': ['FB15K237'],
    'codex_s': ['codex_s'],
    'codex_m': ['codex_m'],
    'codex_l': ['codex_l'],
    'NELL995': ['NELL995'],
    'GDELT': ['GDELT'],
    'ICEWS1819': ['ICEWS1819'],
    'Enron': ['Enron'],
    'Googlemap_CT': ['Googlemap_CT'],
    'scaling_law_1': ['arxiv', 'chempcba', 'FB15K237'],
    'scaling_law_2': ['arxiv', 'chempcba', 'FB15K237', 'products', 'WN18RR'],
    'scaling_law_3': ['arxiv', 'chempcba', 'FB15K237', 'products', 'WN18RR', 'chemblpre', 'arxiv23', 'amazonratings',
                      'NELL995', 'Enron'],
    'scaling_law_4': ['arxiv', 'cora', 'citeseer', 'pubmed', 'arxiv23', 'dblp', 'bookhis', 'bookchild', 'elecomp',
                      'elephoto', 'sportsfit', 'amazonratings', 'products', 'chemblpre', 'chempcba', 'chemhiv', 'bbbp',
                      'bace', 'toxcast', 'cyp450', 'tox21', 'muv', 'WN18RR', 'FB15K237', 'codex_s', 'codex_m',
                      'codex_l', 'NELL995', 'GDELT', 'ICEWS1819', 'Enron', 'Googlemap_CT'],
}
domain2task = {
    'citation': 'node',
    'ecommerce': 'node',
    'kg': 'edge',
    'temporal': 'edge',
    'molecule': 'graph'
}
dataset2domain = {d: 'citation' for d in citation_datasets} | {d: 'ecommerce' for d in ecommerce_datasets} | \
                 {d: 'kg' for d in kg_datasets} | {d: 'molecule' for d in molecule_datasets} | \
                 {d: 'temporal' for d in temporal_datasets}


class VirtualNodeAugmentor:
    def augment(self, data, task):
        assert data.x.ndim == 1, "Node features should be 1D indices"

        if task == 'node':
            return self.add_virtual_nodes_node_classification(data)
        elif task == 'edge':
            return self.add_virtual_nodes_edge_classification(data)
        elif task == 'graph':
            return self.add_virtual_nodes_graph_classification(data)
        else:
            raise ValueError(f"Unknown task: {task}")

    def add_virtual_nodes_node_classification(self, data):
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_nodes) * num_nodes]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes * 2, dtype=torch.long)

        new_edge = torch.tensor([[i, num_nodes + i] for i in range(num_nodes)], dtype=torch.long).t()
        new_edge = to_undirected(new_edge)
        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_edge_classification(self, data):
        num_edges = data.edge_index.size(1)
        num_nodes = data.num_nodes
        node_dim = data.node_text_feat.size(1)

        data.x = torch.cat([data.x, torch.ones(num_edges) * num_nodes]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_edges, dtype=torch.long)

        # Note: This is efficient enough
        new_edge = []
        for i in range(num_edges):
            src, dst = data.edge_index[:, i]
            new_edge.append([src, num_nodes + i])
            new_edge.append([num_nodes + i, dst])
        new_edge = torch.tensor(new_edge, dtype=torch.long).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx

    def add_virtual_nodes_graph_classification(self, data):
        num_nodes = data.x.shape[0]
        num_node_texts = data.node_text_feat.shape[0]
        node_dim = data.node_text_feat.shape[1]

        groups = data.groups  # the group (i.e. graph) index of each node
        num_groups = groups.max() + 1

        data.x = torch.cat([data.x, torch.ones(num_groups) * num_node_texts]).long()
        data.node_text_feat = torch.cat([data.node_text_feat, torch.zeros(1, node_dim)])
        task_node_idx = torch.arange(num_nodes, num_nodes + num_groups, dtype=torch.long)

        i_indices = torch.arange(num_nodes, dtype=torch.long)
        new_edge = torch.stack([i_indices, num_nodes + groups], dim=1).t()
        new_edge = to_undirected(new_edge)

        data.edge_index = torch.cat([data.edge_index, new_edge], dim=1)

        return data, task_node_idx


def preprocess(data):
    dataset_name = data.name
    if dataset_name in citation_datasets + ecommerce_datasets + kg_datasets + temporal_datasets:
        data.x = torch.arange(data.num_nodes)

    elif dataset_name in molecule_datasets:
        data = data.data
        data.edge_index = data.pre_edge_index
        data.node_text_feat = data.node_embs

    return data


def postprocess(data):
    keys = ['x', 'edge_index', 'node_text_feat']
    for k, v in data.to_dict().items():
        if k not in keys:
            data[k] = None
    return data


def preprocess_data_dict(data_dict, task_node_idx_dict):
    x_start = 0
    cnt = 0
    for dataset_name, data in data_dict.items():
        task_node_idx = task_node_idx_dict[dataset_name]

        num_nodes = data.x.shape[0]
        num_unique_nodes = data.node_text_feat.shape[0]

        print(f"Preprocessing {dataset_name} with {num_nodes} nodes and {num_unique_nodes} unique nodes")

        data.x += x_start
        x_start += num_unique_nodes

        task_node_idx += cnt
        cnt += num_nodes

        data_dict[dataset_name] = data
        task_node_idx_dict[dataset_name] = task_node_idx

    return data_dict, task_node_idx_dict


def unified_data(params):
    data_path = params['data_path']
    pre_datasets = pretrain_datasets[params['pretrain_dataset']]

    vn = VirtualNodeAugmentor()

    data_dict = {}
    task_node_idx_dict = {}
    for dataset in pre_datasets:
        data = get_data({'data_path': data_path, 'dataset': dataset, 'task': domain2task[dataset2domain[dataset]]})
        data = preprocess(data)
        data, task_node_idx = vn.augment(data, task=domain2task[dataset2domain[dataset]])
        data = postprocess(data)
        data_dict[dataset] = data
        task_node_idx_dict[dataset] = task_node_idx

    data_dict, task_node_idx_dict = preprocess_data_dict(data_dict, task_node_idx_dict)
    unified_dataset = Batch.from_data_list(list(data_dict.values()))

    return unified_dataset, task_node_idx_dict


# ---------------------------------------------------------------------- #
#  GFM-Safety: multi-dataset pretraining via sgb.data.tag_registry        #
# ---------------------------------------------------------------------- #
#
# The helpers below mirror GFT's get_pt_data / get_train_node_idx pattern
# but source every dataset through our unified TAG data layer (tag_registry)
# and apply GIT's own VirtualNodeAugmentor per dataset so node / link / graph
# tasks can be mixed in one NeighborLoader batch.

def _load_pt_weights(group_name, yaml_path=None):
    """Load per-dataset sampling weights for a pretrain group (e.g. 'all')."""
    import yaml
    if yaml_path is None:
        yaml_path = osp.abspath(
            osp.join(osp.dirname(__file__), "..", "config", "pt_data.yaml")
        )
    with open(yaml_path, "r") as f:
        all_groups = yaml.safe_load(f)
    if group_name not in all_groups:
        raise KeyError(
            f"Pretrain group '{group_name}' not found in {yaml_path}. "
            f"Available: {list(all_groups.keys())}"
        )
    return all_groups[group_name]


def is_pretrain_group(group_name, yaml_path=None):
    """Return True iff group_name is a multi-dataset group defined in pt_data.yaml."""
    import yaml
    if yaml_path is None:
        yaml_path = osp.abspath(
            osp.join(osp.dirname(__file__), "..", "config", "pt_data.yaml")
        )
    if not osp.exists(yaml_path):
        return False
    with open(yaml_path, "r") as f:
        all_groups = yaml.safe_load(f) or {}
    return group_name in all_groups


def unified_data_tag(group_name, yaml_path=None):
    """Build a unified multi-dataset pretrain batch via tag_registry.

    Args:
        group_name: key in pt_data.yaml (e.g. 'all', 'node', 'link', 'graph').
        yaml_path: optional override for pt_data.yaml.

    Returns:
        (unified_dataset, task_node_idx_dict, weights_dict)
          * unified_dataset: PyG Batch of all datasets concatenated
          * task_node_idx_dict: {dataset_name: LongTensor of virtual-node indices
                                into the unified batch}
          * weights_dict: {dataset_name: float weight} preserved in insertion order
    """
    # Project root import shim so we can grab tag_registry without being
    # installed as a package.
    import sys
    _project_root = osp.abspath(
        osp.join(osp.dirname(__file__), "..", "..", "..", "..")
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag

    weights_dict = _load_pt_weights(group_name, yaml_path)
    pre_datasets = list(weights_dict.keys())
    print(f"[GIT] Unified pretrain on {pre_datasets} (group='{group_name}')")

    vn = VirtualNodeAugmentor()
    data_dict = {}
    task_node_idx_dict = {}

    for dataset_name in pre_datasets:
        data, slices = load_tag(dataset_name)

        # tag_registry already renames node_embs/edge_embs/pretrain_edge_index,
        # but be defensive.
        if getattr(data, "node_embs", None) is not None:
            data.node_text_feat = data.node_embs
            data.node_embs = None
        if getattr(data, "edge_embs", None) is not None:
            data.edge_text_feat = data.edge_embs
            data.edge_embs = None
        if getattr(data, "pretrain_edge_index", None) is not None:
            data.edge_index = data.pretrain_edge_index
            data.pretrain_edge_index = None

        # Collapse raw 2D node features down to 1D indices into node_text_feat
        # (GIT's convention — VirtualNodeAugmentor asserts data.x.ndim == 1).
        if data.x.ndim == 2:
            data.x = torch.arange(data.node_text_feat.size(0), dtype=torch.long)

        task = domain2task[dataset2domain[dataset_name]]

        # Molecule data comes as a batch of many small graphs; rebuild the
        # per-node graph-id assignment so the graph-level virtual-node trick
        # in VirtualNodeAugmentor works.
        if task == "graph" and not hasattr(data, "groups"):
            if slices is not None and "x" in slices:
                ptr = slices["x"]
                data.groups = torch.cat([
                    torch.full((ptr[i + 1] - ptr[i],), i, dtype=torch.long)
                    for i in range(len(ptr) - 1)
                ])
            else:
                raise RuntimeError(
                    f"Cannot reconstruct per-graph groups for '{dataset_name}': "
                    "tag_registry returned no slices['x']"
                )

        data, task_node_idx = vn.augment(data, task=task)
        data = postprocess(data)
        data_dict[dataset_name] = data
        task_node_idx_dict[dataset_name] = task_node_idx

    data_dict, task_node_idx_dict = preprocess_data_dict(data_dict, task_node_idx_dict)
    unified_dataset = Batch.from_data_list(list(data_dict.values()))

    return unified_dataset, task_node_idx_dict, weights_dict


def build_weighted_train_nodes(task_node_idx_dict, weights_dict):
    """GFT-style per-dataset weighted oversample/subsample of task node indices.

    For each dataset:
      - integer part N of the weight  -> repeat task_node_idx N times
      - fractional part f             -> sample f * len(task_node_idx) extra
                                         nodes uniformly without replacement
    Returns a single LongTensor of node indices (shuffled by the downstream
    NeighborLoader since we pass shuffle=True).
    """
    chunks = []
    for name, task_idx in task_node_idx_dict.items():
        w = float(weights_dict[name])
        int_w = int(w)
        mod_w = w - int_w

        left = task_idx.repeat(int_w) if int_w > 0 else task_idx.new_empty((0,))
        if mod_w > 0 and task_idx.numel() > 0:
            k = int(mod_w * task_idx.size(0))
            if k > 0:
                perm = torch.randperm(task_idx.size(0))[:k]
                right = task_idx[perm]
            else:
                right = task_idx.new_empty((0,))
        else:
            right = task_idx.new_empty((0,))

        chunks.append(torch.cat([left, right]))
    return torch.cat(chunks) if chunks else torch.empty((0,), dtype=torch.long)

