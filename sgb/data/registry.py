"""Unified dataset registry for GFM-Safety Benchmark.

All datasets are loaded via PyG and returned as a standardized dict:
    {
        "data":    Data or list[Data],
        "splits":  task-specific splits,
        "meta":    {"task": str, "domain": str, "num_classes": int, ...},
    }

No transforms are applied here.
"""

import torch
from torch_geometric.datasets import (Planetoid, Amazon, Flickr, WikiCS, TUDataset, CitationFull,
                                      Reddit, Coauthor, Actor, HeterophilousGraphDataset)
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def _load_cora(root: str) -> dict:
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        "meta": {
            "name": "cora", "task": "node", "domain": "citation",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_pubmed(root: str) -> dict:
    dataset = Planetoid(root=root, name="PubMed")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        "meta": {
            "name": "pubmed", "task": "node", "domain": "biomedical",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_arxiv(root: str) -> dict:
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    data = dataset[0]
    data.y = data.y.squeeze()
    data.edge_index = to_undirected(data.edge_index)
    # OGB uses index splits
    idx = dataset.get_idx_split()
    n = data.num_nodes
    splits = {}
    for name, key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx[key]] = True
        splits[name] = mask
    return {
        "data": data,
        "splits": splits,
        "meta": {
            "name": "arxiv", "task": "node", "domain": "citation",
            "num_classes": dataset.num_classes, "num_features": data.x.size(1),
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
            "node_year": data.node_year.squeeze() if hasattr(data, "node_year") else None,
        },
    }


def _load_wikics(root: str) -> dict:
    dataset = WikiCS(root=root + "/wikics")
    data = dataset[0]
    # WikiCS has 20 splits; use split 0
    return {
        "data": data,
        "splits": {
            "train": data.train_mask[:, 0],
            "val": data.val_mask[:, 0],
            "test": data.test_mask,
        },
        "meta": {
            "name": "wikics", "task": "node", "domain": "wikipedia",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_amazon_photo(root: str) -> dict:
    dataset = Amazon(root=root + "/amazon", name="Photo")
    data = dataset[0]
    # No official split — create random 60/20/20
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "amazon_photo", "task": "node", "domain": "e-commerce",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_citeseer(root: str) -> dict:
    dataset = Planetoid(root=root, name="CiteSeer")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        "meta": {
            "name": "citeseer", "task": "node", "domain": "citation",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_amazon_computers(root: str) -> dict:
    dataset = Amazon(root=root + "/amazon", name="Computers")
    data = dataset[0]
    # No official split — create random 60/20/20
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "amazon_computers", "task": "node", "domain": "e-commerce",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_dblp(root: str) -> dict:
    dataset = CitationFull(root=root + "/dblp", name="DBLP")
    data = dataset[0]
    # No official split — create random 60/20/20
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "dblp", "task": "node", "domain": "citation",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_cora_full(root: str) -> dict:
    dataset = CitationFull(root=root + "/cora_full", name="Cora")
    data = dataset[0]
    # No official split — create random 60/20/20
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "cora_full", "task": "node", "domain": "citation",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_coauthor_cs(root: str) -> dict:
    dataset = Coauthor(root=root + "/coauthor", name="CS")
    data = dataset[0]
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "coauthor_cs", "task": "node", "domain": "academic",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_coauthor_physics(root: str) -> dict:
    dataset = Coauthor(root=root + "/coauthor", name="Physics")
    data = dataset[0]
    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train_mask = torch.zeros(n, dtype=torch.bool); train_mask[perm[:train_end]] = True
    val_mask = torch.zeros(n, dtype=torch.bool); val_mask[perm[train_end:val_end]] = True
    test_mask = torch.zeros(n, dtype=torch.bool); test_mask[perm[val_end:]] = True
    return {
        "data": data,
        "splits": {"train": train_mask, "val": val_mask, "test": test_mask},
        "meta": {
            "name": "coauthor_physics", "task": "node", "domain": "academic",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_actor(root: str) -> dict:
    dataset = Actor(root=root + "/actor")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask[:, 0], "val": data.val_mask[:, 0], "test": data.test_mask[:, 0]},
        "meta": {
            "name": "actor", "task": "node", "domain": "social",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_amazon_ratings(root: str) -> dict:
    dataset = HeterophilousGraphDataset(root=root + "/heterophilous", name="Amazon-Ratings")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask[:, 0], "val": data.val_mask[:, 0], "test": data.test_mask[:, 0]},
        "meta": {
            "name": "amazon_ratings", "task": "node", "domain": "e-commerce",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_reddit(root: str) -> dict:
    dataset = Reddit(root=root + "/reddit")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        "meta": {
            "name": "reddit", "task": "node", "domain": "social",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_products(root: str) -> dict:
    dataset = PygNodePropPredDataset(name="ogbn-products", root=root)
    data = dataset[0]
    data.y = data.y.squeeze()
    idx = dataset.get_idx_split()
    n = data.num_nodes
    splits = {}
    for name, key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx[key]] = True
        splits[name] = mask
    return {
        "data": data,
        "splits": splits,
        "meta": {
            "name": "products", "task": "node", "domain": "e-commerce",
            "num_classes": dataset.num_classes, "num_features": data.x.size(1),
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_flickr(root: str) -> dict:
    dataset = Flickr(root=root + "/flickr")
    data = dataset[0]
    return {
        "data": data,
        "splits": {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask},
        "meta": {
            "name": "flickr", "task": "node", "domain": "social",
            "num_classes": dataset.num_classes, "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


# ---------------------------------------------------------------------------
# Link prediction
# ---------------------------------------------------------------------------

def _load_cora_lp(root: str) -> dict:
    dataset = Planetoid(root=root, name="Cora")
    data = dataset[0]
    # Split edges into train/val/test
    splitter = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = splitter(data)
    return {
        "data": data,
        "splits": {"train": train_data, "val": val_data, "test": test_data},
        "meta": {
            "name": "cora_lp", "task": "link", "domain": "citation",
            "num_features": dataset.num_features,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


def _load_collab(root: str) -> dict:
    dataset = PygLinkPropPredDataset(name="ogbl-collab", root=root)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    return {
        "data": data,
        "splits": split_edge,  # dict with "train"/"valid"/"test", each has "edge"/"edge_neg"
        "meta": {
            "name": "collab", "task": "link", "domain": "academic",
            "num_features": data.x.size(1) if data.x is not None else 0,
            "num_nodes": data.num_nodes, "num_edges": data.edge_index.size(1),
        },
    }


# ---------------------------------------------------------------------------
# Graph classification
# ---------------------------------------------------------------------------

def _load_molhiv(root: str) -> dict:
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root=root)
    idx = dataset.get_idx_split()
    return {
        "data": dataset,
        "splits": {"train": idx["train"], "val": idx["valid"], "test": idx["test"]},
        "meta": {
            "name": "molhiv", "task": "graph", "domain": "molecular",
            "num_classes": dataset.num_tasks,
            "num_features": dataset[0].x.size(1),
            "num_graphs": len(dataset),
        },
    }


def _load_proteins(root: str) -> dict:
    dataset = TUDataset(root=root + "/tudataset", name="PROTEINS")
    n = len(dataset)
    perm = torch.randperm(n)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    return {
        "data": dataset,
        "splits": {
            "train": perm[:train_end],
            "val": perm[train_end:val_end],
            "test": perm[val_end:],
        },
        "meta": {
            "name": "proteins", "task": "graph", "domain": "protein",
            "num_classes": dataset.num_classes,
            "num_features": dataset.num_features,
            "num_graphs": n,
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    # Node classification
    "cora": _load_cora,
    "pubmed": _load_pubmed,
    "arxiv": _load_arxiv,
    "wikics": _load_wikics,
    "citeseer": _load_citeseer,
    "amazon_photo": _load_amazon_photo,
    "amazon_computers": _load_amazon_computers,
    "cora_full": _load_cora_full,
    "dblp": _load_dblp,
    "coauthor_cs": _load_coauthor_cs,
    "coauthor_physics": _load_coauthor_physics,
    "actor": _load_actor,
    "amazon_ratings": _load_amazon_ratings,
    "reddit": _load_reddit,
    "products": _load_products,
    "flickr": _load_flickr,
    # Link prediction
    "cora_lp": _load_cora_lp,
    "collab": _load_collab,
    # Graph classification
    "molhiv": _load_molhiv,
    "proteins": _load_proteins,
}


def load(name: str, root: str = "datasets/PyG") -> dict:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](root)


def available() -> list:
    return sorted(_REGISTRY.keys())
