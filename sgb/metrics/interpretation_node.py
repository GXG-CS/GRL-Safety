"""Node-level subgraph-ablation interpretation (V2).

GraphFramEx-style protocol ported from `interpretation_graph.py`:
  Per test node v:
    1. Take K-hop receptive-field subgraph
    2. For each edge e in receptive field, compute saliency score
    3. Rank edges, pick top-k%
    4. Ablate (remove top-k% edges + endpoints not on path to v)
    5. Compute Fid+ / Fid- / char vs the clean prediction

Three model-agnostic explainers:
  - "grad"      : gradient saliency on x, attributed to edges via endpoint norms
  - "occlusion" : remove each edge, observe prob delta (slow, sanity check)
  - "random"    : random saliency, mandatory baseline

Usage from each method's run_interpretation.py:

    from sgb.metrics.interpretation_node import compute_node_fidelity
    interp = compute_node_fidelity(
        model=trained_model,
        x=feat,                    # [N, d] node features
        edge_index=base_edge_index, # [2, E] base graph
        y=y_labels,                # [N]
        test_idx=test_node_idx,    # [n_test]
        device=device,
        forward_fn=lambda ei, x: model_with_edges(ei, x),  # method-specific
        explainer="grad",          # or "occlusion" or "random"
        topk_list=[0.05, 0.10, 0.20, 0.50],
        target="pred",             # "pred" (clean prediction class) or "true"
        K_hop=2,                   # receptive field
        seed=42,
    )

Output: dict with per-topk Fid+/Fid-/char arrays, plus aggregate stats.

References:
  - Amara et al. GraphFramEx, LoG 2022. https://arxiv.org/abs/2206.09677
  - Ying et al. GNNExplainer, NeurIPS 2019. https://arxiv.org/abs/1903.03894
  - Pope et al. Explainability methods for GCNN, CVPR 2019.
  - GInX-Eval, NeurIPS 2023 W (mandatory random baseline).
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph


# --------------------------------------------------------------------------- #
# Core helpers
# --------------------------------------------------------------------------- #

def _undirected_pair_id(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Map (u, v) and (v, u) to the same canonical id = min*MAX + max.

    Used to deduplicate symmetric edges so top-k removal removes whole pairs.
    """
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    # safe id assuming N < 2**31
    return u.long() * (10 ** 9 + 7) + v.long()


def _k_hop_edges(edge_index: torch.Tensor, target: int,
                 num_nodes: int, K: int = 2):
    """Return the (sub_node_idx, sub_edge_index, sub_to_global_eidx) for K-hop receptive field.
    Uses PyG k_hop_subgraph in 'source_to_target' mode; for undirected graph this
    captures both incoming and outgoing within K hops.
    """
    sub_nodes, sub_ei, _, edge_mask = k_hop_subgraph(
        target, K, edge_index, relabel_nodes=False, num_nodes=num_nodes,
        flow="source_to_target",
    )
    # edge_mask is bool over original edge_index; map to original-edge indices
    sub_eidx = edge_mask.nonzero(as_tuple=False).squeeze(-1)
    return sub_nodes, sub_ei, sub_eidx


def _ablate_edge_index(edge_index: torch.Tensor, edges_to_remove_mask: torch.Tensor):
    """Drop columns of edge_index where mask is True. Returns new edge_index."""
    keep = ~edges_to_remove_mask
    return edge_index[:, keep]


# --------------------------------------------------------------------------- #
# Three explainers (return per-edge-pair score over receptive-field edges)
# --------------------------------------------------------------------------- #

def _explain_random(rf_eidx: torch.Tensor, edge_index: torch.Tensor,
                     seed: int = 42) -> torch.Tensor:
    """Random saliency: uniform over receptive-field edge PAIRS (undirected dedup)."""
    src = edge_index[0, rf_eidx].cpu()
    dst = edge_index[1, rf_eidx].cpu()
    pair_id = _undirected_pair_id(src, dst)
    g = torch.Generator(); g.manual_seed(int(seed))
    pair_unique, inv = torch.unique(pair_id, return_inverse=True)
    pair_score = torch.rand(len(pair_unique), generator=g)
    return pair_score[inv]  # CPU tensor


def _explain_grad(model, forward_fn, x: torch.Tensor, edge_index: torch.Tensor,
                  rf_nodes: torch.Tensor, rf_eidx: torch.Tensor,
                  target_node: int, target_class: int,
                  device) -> torch.Tensor:
    """Gradient saliency: ‖∇x_u logit_y(target_node)‖ + ‖∇x_v logit_y(target_node)‖.

    Returns: score per receptive-field edge (length == len(rf_eidx)).
    """
    model.eval()
    x_grad = x.detach().clone().requires_grad_(True)
    logits = forward_fn(edge_index, x_grad)
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    target_logit = logits[target_node, target_class]
    grad = torch.autograd.grad(target_logit, x_grad, retain_graph=False)[0]
    node_score = grad.detach().abs().sum(dim=1)  # [N]

    src = edge_index[0, rf_eidx]
    dst = edge_index[1, rf_eidx]
    edge_score = node_score[src] + node_score[dst]
    return edge_score.cpu()


def _explain_occlusion(model, forward_fn, x: torch.Tensor, edge_index: torch.Tensor,
                        rf_eidx: torch.Tensor, target_node: int, target_class: int,
                        device, max_edges: int = 200) -> torch.Tensor:
    """Edge occlusion: drop each edge-pair, measure prob delta on target_node.

    SLOW: O(E_rf) forwards. For large receptive fields, subsample to max_edges.
    """
    model.eval()
    src = edge_index[0, rf_eidx]
    dst = edge_index[1, rf_eidx]
    pair_id = _undirected_pair_id(src, dst)
    pair_unique, inv = torch.unique(pair_id, return_inverse=True)
    n_pairs = len(pair_unique)

    if n_pairs > max_edges:
        sel = torch.randperm(n_pairs)[:max_edges]
        # other pairs get score 0 (least important by default)
    else:
        sel = torch.arange(n_pairs)

    with torch.no_grad():
        clean_logits = forward_fn(edge_index, x)
        if clean_logits.dim() == 3:
            clean_logits = clean_logits.squeeze(0)
        clean_prob = torch.softmax(clean_logits[target_node], dim=0)[target_class].item()

    pair_score = torch.zeros(n_pairs)
    for idx_in_sel in sel.tolist():
        pair_mask = inv == idx_in_sel
        rm_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        rm_mask[rf_eidx[pair_mask]] = True
        ei_mod = _ablate_edge_index(edge_index, rm_mask)
        with torch.no_grad():
            logits_mod = forward_fn(ei_mod, x)
            if logits_mod.dim() == 3:
                logits_mod = logits_mod.squeeze(0)
            prob_mod = torch.softmax(logits_mod[target_node], dim=0)[target_class].item()
        pair_score[idx_in_sel] = clean_prob - prob_mod  # higher = more important

    # Distribute back to per-edge scores (each edge inherits its pair's score)
    return pair_score[inv]


# --------------------------------------------------------------------------- #
# Fidelity computation for a single test node, all topk simultaneously
# --------------------------------------------------------------------------- #

def _char_score(fp: float, fm: float) -> float:
    fp_c = max(0.0, min(1.0, fp))
    fm_c = max(0.0, min(1.0, fm))
    num = 2.0 * fp_c * (1.0 - fm_c)
    denom = fp_c + (1.0 - fm_c) + 1e-8
    return num / denom


def _fidelity_one_node(forward_fn, x, edge_index, target_node, target_class,
                        rf_eidx, edge_score, topk_list):
    """Given per-edge score, compute Fid+/Fid-/char at each topk fraction.

    Top-k% edges are RANKED BY edge_score (descending). We deduplicate
    undirected pairs so removing 1 pair removes both (u,v) and (v,u) edges.
    """
    # Force CPU for ranking (small tensors, avoids device mismatch with mixed sources)
    src = edge_index[0, rf_eidx].cpu()
    dst = edge_index[1, rf_eidx].cpu()
    pair_id = _undirected_pair_id(src, dst)
    pair_unique, inv = torch.unique(pair_id, return_inverse=True)
    edge_score_cpu = edge_score.detach().cpu().float()
    # pair score = max within pair
    pair_score = torch.full((len(pair_unique),), -float("inf"))
    pair_score = pair_score.scatter_reduce(
        0, inv, edge_score_cpu, reduce="amax", include_self=False)
    n_pairs = len(pair_unique)
    if n_pairs == 0:
        return None

    sorted_pair = torch.argsort(pair_score, descending=True)

    # clean baseline prob
    with torch.no_grad():
        clean_logits = forward_fn(edge_index, x)
        if clean_logits.dim() == 3:
            clean_logits = clean_logits.squeeze(0)
        clean_prob = torch.softmax(clean_logits[target_node], dim=0)[target_class].item()

    rf_eidx_cpu = rf_eidx.detach().cpu()
    E_total = edge_index.size(1)

    out = {}
    for topk in topk_list:
        n_keep = max(1, int(round(n_pairs * topk)))
        top_pair_local = sorted_pair[:n_keep]

        # Fid+ : remove top-k% edges -> prob drop
        rm_mask_plus = torch.zeros(E_total, dtype=torch.bool)
        in_top = torch.isin(inv, top_pair_local)  # bool over rf edges (CPU)
        rm_mask_plus[rf_eidx_cpu[in_top]] = True
        rm_mask_plus_dev = rm_mask_plus.to(edge_index.device)
        ei_plus = _ablate_edge_index(edge_index, rm_mask_plus_dev)
        with torch.no_grad():
            logits_p = forward_fn(ei_plus, x)
            if logits_p.dim() == 3:
                logits_p = logits_p.squeeze(0)
            prob_plus = torch.softmax(logits_p[target_node], dim=0)[target_class].item()
        fid_plus = clean_prob - prob_plus  # higher = better explanation

        # Fid- : keep ONLY top-k% receptive-field edges of the FULL graph
        not_in_top = ~in_top
        rm_global = torch.zeros(E_total, dtype=torch.bool)
        rm_global[rf_eidx_cpu[not_in_top]] = True
        rm_global_dev = rm_global.to(edge_index.device)
        ei_minus = _ablate_edge_index(edge_index, rm_global_dev)
        with torch.no_grad():
            logits_m = forward_fn(ei_minus, x)
            if logits_m.dim() == 3:
                logits_m = logits_m.squeeze(0)
            prob_minus = torch.softmax(logits_m[target_node], dim=0)[target_class].item()
        fid_minus = clean_prob - prob_minus  # lower = better (explanation alone is sufficient)

        char = _char_score(fid_plus, fid_minus)
        out[topk] = {
            "fid_plus": fid_plus,
            "fid_minus": fid_minus,
            "char": char,
            "clean_prob": clean_prob,
            "n_pairs_in_rf": int(n_pairs),
            "n_removed": int(n_keep),
        }
    return out


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #

def compute_node_fidelity(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    test_idx,                   # iterable of node idx
    device,
    forward_fn,                 # callable(edge_index, x) -> logits[N, C]
    explainer: str = "grad",    # "grad" | "occlusion" | "random"
    topk_list=(0.05, 0.10, 0.20, 0.50),
    target: str = "pred",       # "pred" | "true"
    K_hop: int = 2,
    seed: int = 42,
    occlusion_max_edges: int = 200,
    verbose: bool = False,
):
    """Run subgraph-ablation interpretation over `test_idx` test nodes.

    Returns a dict:
        {
          "per_topk": {topk: {fid_plus_mean, fid_plus_std,
                              fid_minus_mean, fid_minus_std,
                              char_mean, char_std, n_eval}},
          "explainer": str,
          "n_test_attempted": int,
          "n_test_used": int,
          "skipped_no_rf": int,
          "skipped_pred_wrong": int,
        }
    """
    model.eval()
    test_idx = torch.as_tensor(test_idx, dtype=torch.long).cpu()

    # Pre-compute clean predictions once
    with torch.no_grad():
        clean_logits_all = forward_fn(edge_index, x)
        if clean_logits_all.dim() == 3:
            clean_logits_all = clean_logits_all.squeeze(0)
        clean_pred = clean_logits_all.argmax(dim=-1).cpu()

    per_topk_records = {topk: [] for topk in topk_list}
    skip_no_rf = 0
    skip_pred_wrong = 0
    n_used = 0

    for i, v in enumerate(test_idx.tolist()):
        if target == "pred":
            cls = int(clean_pred[v].item())
        else:
            cls = int(y[v].item())
            if cls != int(clean_pred[v].item()):
                skip_pred_wrong += 1
                # for "true" target on wrongly-classified nodes, keep going,
                # since explanation can still be measured w.r.t. true class
                pass

        rf_nodes, _, rf_eidx = _k_hop_edges(edge_index, v, x.size(0), K=K_hop)
        if len(rf_eidx) == 0:
            skip_no_rf += 1
            continue

        # Compute per-edge saliency
        if explainer == "random":
            edge_score = _explain_random(rf_eidx, edge_index, seed=seed + i)
        elif explainer == "grad":
            edge_score = _explain_grad(model, forward_fn, x, edge_index,
                                        rf_nodes, rf_eidx, v, cls, device)
        elif explainer == "occlusion":
            edge_score = _explain_occlusion(model, forward_fn, x, edge_index,
                                             rf_eidx, v, cls, device,
                                             max_edges=occlusion_max_edges)
        else:
            raise ValueError(f"Unknown explainer: {explainer}")

        node_records = _fidelity_one_node(
            forward_fn, x, edge_index, v, cls, rf_eidx, edge_score, list(topk_list))
        if node_records is None:
            skip_no_rf += 1
            continue
        for topk, rec in node_records.items():
            per_topk_records[topk].append(rec)
        n_used += 1
        if verbose and (i + 1) % 25 == 0:
            print(f"  processed {i + 1}/{len(test_idx)} (used {n_used})")

    # Aggregate
    per_topk_agg = {}
    for topk, recs in per_topk_records.items():
        if not recs:
            per_topk_agg[topk] = {"n_eval": 0}
            continue
        fp = np.array([r["fid_plus"] for r in recs])
        fm = np.array([r["fid_minus"] for r in recs])
        ch = np.array([r["char"] for r in recs])
        per_topk_agg[topk] = {
            "fid_plus_mean": float(fp.mean()),
            "fid_plus_std": float(fp.std(ddof=1) if len(fp) > 1 else 0.0),
            "fid_minus_mean": float(fm.mean()),
            "fid_minus_std": float(fm.std(ddof=1) if len(fm) > 1 else 0.0),
            "char_mean": float(ch.mean()),
            "char_std": float(ch.std(ddof=1) if len(ch) > 1 else 0.0),
            "n_eval": int(len(recs)),
        }
    return {
        "per_topk": per_topk_agg,
        "explainer": explainer,
        "n_test_attempted": int(len(test_idx)),
        "n_test_used": int(n_used),
        "skipped_no_rf": int(skip_no_rf),
        "skipped_pred_wrong": int(skip_pred_wrong),
        "target": target,
    }
