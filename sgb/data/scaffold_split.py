"""Bemis-Murcko scaffold split builder for BACE / Tox21.

The TAG cache (`datasets/TAG/<ds>/processed/geometric_data_processed.pt`) does
not match OGB graph order: the OFA pipeline canonical-reorders atoms differently
from OGB's smiles2graph. To get scaffolds aligned to TAG indices we:

  1. Build a per-graph fingerprint from TAG (n_atoms, label, sorted-edge-set).
  2. Build the same fingerprint from OGB graphs.
  3. Greedy 1-1 match TAG_idx -> OGB_idx by fingerprint.
  4. Use OGB SMILES at the matched index to compute Bemis-Murcko scaffold for
     each TAG index. Ambiguous fingerprints (~10-25%) match to any OGB index
     with the same fingerprint; for those the scaffold may correspond to a
     near-isomorphic molecule, which is still a valid scaffold proxy.

Output (cached):
    {"random": {"train","val","test"},
     "scaffold": {"train","val","test"},
     "smiles": list[str],         # OGB SMILES at the matched OGB index
     "scaffolds": list[str],      # Bemis-Murcko scaffold per TAG index
     "n": int,
     "n_unique_scaffolds": int,
     "n_ambiguous_matches": int,
     "dataset": str}
"""
from __future__ import annotations

import os
import os.path as osp
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_DEFAULT_ROOT = osp.join(_PROJECT_ROOT, "datasets")
_CACHE_ROOT = osp.join(_PROJECT_ROOT, "cache_data", "scaffold_splits")

_OGB_NAME = {
    "bace": "ogbg-molbace",
    "tox21": "ogbg-moltox21",
}


def _bemis_murcko(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


def _edge_fp(edge_index: torch.Tensor) -> tuple:
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    return tuple(sorted([(min(a, b), max(a, b)) for a, b in edges]))


def _tag_fingerprints(dataset_name: str) -> List[tuple]:
    p = osp.join(_PROJECT_ROOT, "datasets", "TAG", dataset_name,
                 "processed", "geometric_data_processed.pt")
    g, sliced = torch.load(p, map_location="cpu", weights_only=False)
    n_graphs = len(sliced["y"]) - 1
    fps = []
    for i in range(n_graphs):
        n_atoms = sliced["x"][i + 1].item() - sliced["x"][i].item()
        es, ee = sliced["edge_index"][i].item(), sliced["edge_index"][i + 1].item()
        ei = g.edge_index[:, es:ee]
        ys, ye = sliced["y"][i].item(), sliced["y"][i + 1].item()
        y_vec = g.y[ys:ye].view(-1)
        # Use first label (binary task) for matching; isnan -> -1 sentinel
        if y_vec.dtype.is_floating_point and torch.isnan(y_vec[0]):
            y = -1
        else:
            y = int(y_vec[0].item())
        fps.append((int(n_atoms), y, _edge_fp(ei)))
    return fps, int(n_graphs)


def _ogb_smiles_and_fps(dataset_name: str, root: str):
    import pandas as pd
    from ogb.graphproppred import PygGraphPropPredDataset

    name = _OGB_NAME[dataset_name]
    ds = PygGraphPropPredDataset(name=name, root=root)
    folder = osp.join(root, name.replace("-", "_"), "mapping", "mol.csv.gz")
    df = pd.read_csv(folder, compression="gzip")
    smiles = df["smiles"].astype(str).tolist()

    fps = []
    for i in range(len(ds)):
        d = ds[i]
        y_vec = d.y.view(-1)
        if y_vec.dtype.is_floating_point and torch.isnan(y_vec[0]):
            y = -1
        else:
            y = int(y_vec[0].item())
        fps.append((int(d.num_nodes), y, _edge_fp(d.edge_index)))
    assert len(smiles) == len(fps), f"{len(smiles)} vs {len(fps)}"
    return smiles, fps


def _greedy_match(tag_fps: List[tuple], ogb_fps: List[tuple]) -> Tuple[List[int], int]:
    """Map each TAG index to a unique OGB index sharing the same fingerprint."""
    bucket: Dict[tuple, List[int]] = defaultdict(list)
    for j, fp in enumerate(ogb_fps):
        bucket[fp].append(j)
    used = [False] * len(ogb_fps)
    mapping = [-1] * len(tag_fps)
    n_ambig = 0
    for i, fp in enumerate(tag_fps):
        candidates = bucket.get(fp, [])
        chosen = -1
        for j in candidates:
            if not used[j]:
                chosen = j
                used[j] = True
                break
        if chosen == -1:
            raise RuntimeError(f"TAG index {i} has fp not in OGB: {fp[:2]}")
        mapping[i] = chosen
        if len(candidates) > 1:
            n_ambig += 1
    return mapping, n_ambig


def _scaffold_split(scaffolds: List[str], n: int,
                    train_frac: float = 0.8,
                    val_frac: float = 0.1) -> Dict[str, np.ndarray]:
    clusters: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        clusters[s].append(i)
    sorted_clusters = sorted(clusters.values(), key=lambda x: (-len(x), x[0]))

    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for cl in sorted_clusters:
        if len(train) + len(cl) <= n_train:
            train.extend(cl)
        elif len(val) + len(cl) <= n_val:
            val.extend(cl)
        else:
            test.extend(cl)

    return {
        "train": np.asarray(sorted(train), dtype=np.int64),
        "val": np.asarray(sorted(val), dtype=np.int64),
        "test": np.asarray(sorted(test), dtype=np.int64),
    }


def _random_split(n: int, train_frac: float = 0.8,
                  val_frac: float = 0.1, seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    return {
        "train": np.sort(perm[:n_train]).astype(np.int64),
        "val": np.sort(perm[n_train:n_train + n_val]).astype(np.int64),
        "test": np.sort(perm[n_train + n_val:]).astype(np.int64),
    }


def load_splits(dataset_name: str, n_graphs: int,
                root: str = None, force: bool = False) -> dict:
    if dataset_name not in _OGB_NAME:
        raise ValueError(f"scaffold split only defined for {list(_OGB_NAME)}, got {dataset_name}")
    root = root or _DEFAULT_ROOT
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    cache_path = osp.join(_CACHE_ROOT, f"{dataset_name}.pt")

    if osp.exists(cache_path) and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    tag_fps, n_tag = _tag_fingerprints(dataset_name)
    if n_tag != n_graphs:
        raise RuntimeError(f"TAG cache has {n_tag} graphs, expected {n_graphs}")

    smiles, ogb_fps = _ogb_smiles_and_fps(dataset_name, root)
    if len(smiles) < n_graphs:
        raise RuntimeError(
            f"OGB SMILES count {len(smiles)} < TAG cache count {n_graphs}"
        )
    if len(smiles) != n_graphs:
        print(f"[scaffold_split] {dataset_name}: OGB has {len(smiles)} mols, "
              f"TAG has {n_graphs}; will match each TAG mol to an OGB twin, "
              f"leaving {len(smiles) - n_graphs} OGB mols unmatched.")

    mapping, n_ambig = _greedy_match(tag_fps, ogb_fps)
    aligned_smiles = [smiles[j] for j in mapping]
    scaffolds = [_bemis_murcko(s) for s in aligned_smiles]
    n_unique = len(set(scaffolds))

    print(f"[scaffold_split] {dataset_name}: matched {n_graphs} graphs, "
          f"{n_ambig} via ambiguous fingerprint, "
          f"{n_unique} unique scaffolds")

    out = {
        "random": {k: torch.from_numpy(v) for k, v in _random_split(n_graphs, seed=0).items()},
        "scaffold": {k: torch.from_numpy(v) for k, v in _scaffold_split(scaffolds, n_graphs).items()},
        "smiles": aligned_smiles,
        "scaffolds": scaffolds,
        "n": n_graphs,
        "n_unique_scaffolds": n_unique,
        "n_ambiguous_matches": n_ambig,
        "dataset": dataset_name,
    }
    torch.save(out, cache_path)
    return out


if __name__ == "__main__":
    for ds, n in [("bace", 1513), ("tox21", 7831)]:
        r = load_splits(ds, n, force=True)
        sc, rd = r["scaffold"], r["random"]
        print(f"=== {ds} (n={n}, unique scaffolds={r['n_unique_scaffolds']}, "
              f"ambig matches={r['n_ambiguous_matches']}) ===")
        for name, sp in [("scaffold", sc), ("random", rd)]:
            print(f"  {name:8s} train={len(sp['train']):4d} "
                  f"val={len(sp['val']):4d} test={len(sp['test']):4d}")
