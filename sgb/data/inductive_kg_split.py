"""Build inductive KG splits on our own TAG cache.

Following Teru+2020 (GraIL) protocol but using our own data and SBERT 768d
node features for paper-internal consistency. We do NOT download the GraIL
public splits.

Procedure:
  1. Random entity partition E_tr / E_te (75/25 by default).
  2. G_tr = edges with both endpoints in E_tr.
     G_te = edges with both endpoints in E_te.
     Cross-edges discarded.
  3. Within G_te, randomly split edges 80/10/10:
       G_te_support  (eval-time message-passing graph)
       Q_te_valid    (inductive validation queries)
       Q_te_test     (final inductive test queries)
  4. Relation-overlap constraint: drop any query edge in Q_te_{valid,test}
     whose relation is not present in G_tr (avoid zero-shot relation, OOS).
     Constraint applies to queries only; G_te_support may keep zero-shot
     relations (encoder doesn't read relation type, so harmless).
  5. If query drop ratio > 5%, resample partition seed (up to 10 retries).

Returns a dict with PyG-style index tensors and entity feature subsets.
"""
from __future__ import annotations

import os
import os.path as osp
from typing import Dict

import numpy as np
import torch

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
_CACHE_ROOT = osp.join(_PROJECT_ROOT, "cache_data", "inductive_kg")


def _load_full_kg(dataset_name: str):
    p = osp.join(_PROJECT_ROOT, "datasets", "TAG", dataset_name,
                 "processed", "geometric_data_processed.pt")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    g = obj[0] if isinstance(obj, tuple) else obj
    feat = g.node_text_feat.float()  # [N, 768]
    return {
        "edge_index": g.edge_index.long(),       # [2, E_total]
        "edge_types": g.edge_types.long(),       # [E_total]
        "node_text_feat": feat,                  # [N, 768]
        "num_nodes": int(g.num_nodes),
    }


def _build_inductive_split(kg, partition_seed: int, test_frac: float = 0.25,
                           query_frac: float = 0.10, rel_drop_tol: float = 0.05):
    """One attempt; returns dict or raises if drop ratio exceeds tolerance."""
    rng = np.random.RandomState(partition_seed)
    N = kg["num_nodes"]
    E_total = kg["edge_index"].size(1)
    edge_index = kg["edge_index"].numpy()
    edge_types = kg["edge_types"].numpy()

    # Step 1: entity partition
    perm = rng.permutation(N)
    n_te = int(round(test_frac * N))
    E_te_set = perm[:n_te]
    E_tr_set = perm[n_te:]
    is_te = np.zeros(N, dtype=bool); is_te[E_te_set] = True
    is_tr = np.zeros(N, dtype=bool); is_tr[E_tr_set] = True

    # Step 2: edge classification
    h, t = edge_index[0], edge_index[1]
    in_tr = is_tr[h] & is_tr[t]
    in_te = is_te[h] & is_te[t]
    # crossing = ~(in_tr | in_te)  # discarded

    G_tr_eidx = np.flatnonzero(in_tr)
    G_te_eidx = np.flatnonzero(in_te)

    if len(G_te_eidx) < 50:
        raise RuntimeError(f"G_te too small: {len(G_te_eidx)} edges")

    # Step 3: split G_te edges 80/10/10
    rng2 = np.random.RandomState(partition_seed + 1)
    perm_te = rng2.permutation(len(G_te_eidx))
    n_q_total = int(round(2 * query_frac * len(G_te_eidx)))  # 20% queries
    n_valid = n_q_total // 2
    support_eidx = G_te_eidx[perm_te[n_q_total:]]
    valid_eidx_pre = G_te_eidx[perm_te[:n_valid]]
    test_eidx_pre = G_te_eidx[perm_te[n_valid:n_q_total]]

    # Step 4: relation-overlap constraint on queries
    rels_in_tr = set(edge_types[G_tr_eidx].tolist())

    def _filter(eidx):
        rels = edge_types[eidx]
        keep_mask = np.array([r in rels_in_tr for r in rels])
        return eidx[keep_mask], int((~keep_mask).sum())

    valid_eidx, n_drop_v = _filter(valid_eidx_pre)
    test_eidx, n_drop_t = _filter(test_eidx_pre)
    n_drop = n_drop_v + n_drop_t
    drop_ratio = n_drop / max(1, n_q_total)
    if drop_ratio > rel_drop_tol:
        raise RuntimeError(f"query drop ratio {drop_ratio:.3f} > {rel_drop_tol}")

    # Build remap: original entity id -> compact id within E_tr or E_te
    # Train graph node features and edges live in original-id space; runner
    # will subset by E_tr_set / E_te_set as needed. We expose both sets.

    return {
        "dataset": None,  # filled by caller
        "partition_seed": partition_seed,
        "num_nodes_full": N,
        "node_text_feat": kg["node_text_feat"],          # [N, 768], full
        "edge_index_full": kg["edge_index"],
        "edge_types_full": kg["edge_types"],

        "E_tr": torch.from_numpy(E_tr_set).long(),       # original entity ids
        "E_te": torch.from_numpy(E_te_set).long(),
        "is_tr": torch.from_numpy(is_tr),
        "is_te": torch.from_numpy(is_te),

        # Train graph (entity ids in original [0, N) space)
        "G_tr_edge_index": kg["edge_index"][:, G_tr_eidx],
        "G_tr_edge_types": kg["edge_types"][G_tr_eidx],
        # Test support graph
        "G_te_support_edge_index": kg["edge_index"][:, support_eidx],
        "G_te_support_edge_types": kg["edge_types"][support_eidx],
        # Inductive valid / test queries
        "Q_te_valid_edge_index": kg["edge_index"][:, valid_eidx],
        "Q_te_valid_edge_types": kg["edge_types"][valid_eidx],
        "Q_te_test_edge_index": kg["edge_index"][:, test_eidx],
        "Q_te_test_edge_types": kg["edge_types"][test_eidx],

        "stats": {
            "n_E_tr": int(is_tr.sum()),
            "n_E_te": int(is_te.sum()),
            "n_G_tr": int(in_tr.sum()),
            "n_G_te_support": int(len(support_eidx)),
            "n_Q_te_valid": int(len(valid_eidx)),
            "n_Q_te_test": int(len(test_eidx)),
            "n_dropped_query": int(n_drop),
            "drop_ratio": float(drop_ratio),
            "n_relations_in_tr": int(len(rels_in_tr)),
            "n_relations_total": int(edge_types.max() + 1),
        },
    }


def load_inductive_split(dataset_name: str, partition_seed: int = 0,
                         force: bool = False, max_retries: int = 10) -> Dict:
    """Load (or build + cache) an inductive split. Retries with seed bumped
    if relation-overlap drop ratio exceeds tolerance."""
    os.makedirs(_CACHE_ROOT, exist_ok=True)
    cache_path = osp.join(_CACHE_ROOT, f"{dataset_name}_seed{partition_seed}.pt")
    if osp.exists(cache_path) and not force:
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    kg = _load_full_kg(dataset_name)

    last_err = None
    for retry in range(max_retries):
        seed = partition_seed * 1000 + retry  # deterministic bump
        try:
            split = _build_inductive_split(kg, partition_seed=seed)
            split["dataset"] = dataset_name
            split["effective_seed"] = seed
            torch.save(split, cache_path)
            return split
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to build inductive split for {dataset_name} after "
        f"{max_retries} retries (last error: {last_err})"
    )


if __name__ == "__main__":
    for ds in ["FB15K237", "WN18RR"]:
        print(f"=== {ds} ===")
        for ps in [0, 1, 2]:
            sp = load_inductive_split(ds, partition_seed=ps, force=True)
            s = sp["stats"]
            print(f"  seed={ps} (eff={sp['effective_seed']}): "
                  f"|E_tr|={s['n_E_tr']} |E_te|={s['n_E_te']} "
                  f"|G_tr|={s['n_G_tr']} |G_te_sup|={s['n_G_te_support']} "
                  f"|Q_v|={s['n_Q_te_valid']} |Q_t|={s['n_Q_te_test']} "
                  f"drop={s['drop_ratio']:.3f} "
                  f"rels_in_tr={s['n_relations_in_tr']}/{s['n_relations_total']}")
