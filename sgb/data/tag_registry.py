"""Unified TAG data registry for GRL-Safety Benchmark.

Thin wrapper over GFT's original OFA data module (sgb/data/dataset/).
Processed data cached at datasets/TAG/{name}/processed/geometric_data_processed.pt

Covers the 25 evaluation datasets used in the paper plus the two
pretraining-only molecular corpora (chemblpre, chempcba):
  Node Classification (15): cora, citeseer, pubmed, arxiv, arxiv23, arxivyear,
                            dblp, wikics, tolokers, elecomp, elephoto,
                            sportsfit, amazonratings, bookhis, bookchild
  Link Prediction (3):      WN18RR, FB15K237 (KG); ml1m (recommendation)
  Graph Classification (7): bace, bbbp, chemhiv, cyp450, muv, tox21, toxcast
  Pretrain-only (2):        chemblpre, chempcba

Usage:
    from sgb.data.tag_registry import load
    data = load("cora")  # auto-prepare on first call
"""

import os
import os.path as osp
import sys
import subprocess

import torch

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", ".."))
_CACHE_ROOT = osp.join(_PROJECT_ROOT, "datasets", "TAG")
_PREPARE_SCRIPT = osp.join(_HERE, "dataset", "prepare_dataset.py")

DATASETS = {
    "cora":     {"task": "node"},
    "citeseer": {"task": "node"},
    "pubmed":   {"task": "node"},
    "wikics":   {"task": "node"},
    "arxiv":    {"task": "node"},
    "elephoto": {"task": "node"},
    "elecomp":  {"task": "node"},
    "tolokers": {"task": "node"},
    "dblp":     {"task": "node"},
    "arxiv23":  {"task": "node"},
    "arxivyear": {"task": "node"},
    "amazonratings": {"task": "node"},
    "bookhis":  {"task": "node"},
    "bookchild": {"task": "node"},
    "sportsfit": {"task": "node"},
    "products": {"task": "node"},
    "goodreads": {"task": "link"},
    "protein_hs": {"task": "link"},
    "ml1m":     {"task": "link"},
    "WN18RR":   {"task": "link"},
    "FB15K237": {"task": "link"},
    "chemhiv":  {"task": "graph"},
    "chempcba": {"task": "graph"},
    "chemblpre": {"task": "graph"},
    # Added from TSGFM
    "tox21":    {"task": "graph"},
    "bace":     {"task": "graph"},
    "bbbp":     {"task": "graph"},
    "muv":      {"task": "graph"},
    "toxcast":  {"task": "graph"},
    "cyp450":   {"task": "graph"},
    "esol":     {"task": "graph"},
    "freesolv": {"task": "graph"},
    "lipo":     {"task": "graph"},
    "molproperties": {"task": "graph"},
    "ml1m_cls": {"task": "link"},
    "expla_graph": {"task": "graph"},
    "scene_graph": {"task": "graph"},
    "wiki_graph": {"task": "graph"},
    "webqsp":   {"task": "graph"},
    "ultrachat200k": {"task": "graph"},
    "mag240m":  {"task": "node"},
    "wikikg90m": {"task": "link"},

    # Link-prediction aliases: reuse the same underlying graph as the
    # corresponding NC dataset, but the task becomes "predict citation edges".
    # These load byte-identical data to their NC counterpart; the caller is
    # expected to use a link-prediction eval on top.
    "cora_link":     {"task": "link", "alias": "cora"},
    "citeseer_link": {"task": "link", "alias": "citeseer"},
    "pubmed_link":   {"task": "link", "alias": "pubmed"},
    "wikics_link":   {"task": "link", "alias": "wikics"},
    "dblp_link":     {"task": "link", "alias": "dblp"},
    "arxiv_link":    {"task": "link", "alias": "arxiv"},
}


def available():
    return sorted(DATASETS.keys())


def load(name, cache_root=None):
    """Load a dataset. Auto-prepares on first call.

    Returns:
        (data, slices) tuple — same format as GFT's OFAPygDataset output.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {available()}")

    # Link-prediction alias: redirect to the underlying NC dataset's cache
    # file and let the caller apply a link-prediction eval on top.
    info = DATASETS[name]
    if "alias" in info:
        name = info["alias"]

    root = cache_root or _CACHE_ROOT
    out_path = osp.join(root, name, "processed", "geometric_data_processed.pt")

    if not osp.exists(out_path):
        print(f"[tag_registry] Cache not found for '{name}', preparing...")
        _prepare(name, root)

    data, slices = torch.load(out_path, weights_only=False)

    # Normalize chem datasets so all 9 datasets expose the same field names.
    # Chem data comes from OFA's molecule pipeline and stores features under
    # node_embs/edge_embs; rename to node_text_feat/edge_text_feat to match
    # the node/link datasets. Also swap in pretrain_edge_index if present.
    if getattr(data, "node_embs", None) is not None:
        data.node_text_feat = data.node_embs
        data.node_embs = None
    if getattr(data, "edge_embs", None) is not None:
        data.edge_text_feat = data.edge_embs
        data.edge_embs = None
    if getattr(data, "pretrain_edge_index", None) is not None:
        data.edge_index = data.pretrain_edge_index
        data.pretrain_edge_index = None

    # Fallback split generation: if the dataset has no train/val/test masks
    # (e.g. arxivyear, ml1m, protein_hs which only ship labels + graph), build
    # deterministic random 60/20/20 masks per node (or per edge for link tasks)
    # so downstream finetune scripts don't see empty training sets and produce
    # the all-zero metric artefact observed in job 24025045/24025046.
    _needs_node_masks = (
        info.get("task") == "node"
        and not hasattr(data, "train_mask")
        and not hasattr(data, "train_masks")
    )
    if _needs_node_masks and hasattr(data, "y") and data.y is not None:
        n = data.y.shape[0]
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n, generator=g)
        n_train = int(0.6 * n); n_val = int(0.2 * n)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train:n_train + n_val]
        test_idx  = perm[n_train + n_val:]
        train_mask = torch.zeros(n, dtype=torch.bool); train_mask[train_idx] = True
        val_mask   = torch.zeros(n, dtype=torch.bool); val_mask[val_idx]     = True
        test_mask  = torch.zeros(n, dtype=torch.bool); test_mask[test_idx]   = True
        data.train_mask = train_mask
        data.val_mask   = val_mask
        data.test_mask  = test_mask
        # Also populate 5-way mask list so paired-seed protocols work
        data.train_masks = [train_mask] * 5
        data.val_masks   = [val_mask] * 5
        data.test_masks  = [test_mask] * 5

    _needs_link_idx = (
        info.get("task") == "link"
        and not hasattr(data, "train_idx")
    )
    if _needs_link_idx and hasattr(data, "y") and data.y is not None:
        n_edges = data.y.shape[0]
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_edges, generator=g)
        n_train = int(0.6 * n_edges); n_val = int(0.2 * n_edges)
        data.train_idx = perm[:n_train]
        data.val_idx   = perm[n_train:n_train + n_val]
        data.test_idx  = perm[n_train + n_val:]

    return data, slices


def load_data(name, cache_root=None):
    """Convenience: load and return just the Data object (no slices)."""
    data, _ = load(name, cache_root)
    return data


def _prepare(name, root):
    """Generate processed .pt by calling prepare_dataset.py in a subprocess.

    Subprocess avoids 'data' module name conflicts between sgb/data/
    and OFA's internal data/ package.
    """
    result = subprocess.run(
        [sys.executable, _PREPARE_SCRIPT, "--name", name, "--root", root],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0:
        print(result.stderr, end="")
        raise RuntimeError(f"[tag_registry] Prepare failed for '{name}'")


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare TAG datasets.")
    parser.add_argument("--prepare", type=str, help="Dataset name or 'all'.")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available:", available())
    elif args.prepare:
        if args.prepare == "all":
            for name in available():
                try:
                    load(name)
                    print(f"  OK: {name}")
                except Exception as e:
                    print(f"  FAILED {name}: {e}")
        else:
            load(args.prepare)
    else:
        parser.print_help()
