"""GFM-Safety graph-level interpretation v2.

Atom-level fidelity for graph classification, mirroring node-level v1
protocol (sgb/metrics/interpretation_v1.py):

  - Saliency: |grad of target-class logit w.r.t. atom features| L1-sum
  - Mask = node ablation (remove atom + incident bonds + drop from pool)
    NOT feature zeroing (avoids OOD; rejected v0 mistake).
  - Fid+ = clean_prob - prob(top-k atoms removed). Higher = better.
  - Fid- = clean_prob - prob(bottom-k atoms removed). LOWER = better.
  - char = 2*Fp*(1-Fm)/(Fp + 1-Fm), GraphFramEx harmonic.
  - Mandatory random baseline: char_random.
  - HEADLINE METRIC: delta_char = char_saliency - char_random.

Eval set policy: only mols with y=1 AND pred(clean)=1 (clean-correct
positives). Report n_eval per cell.

Refs:
  - GraphFramEx, Amara et al. LoG 2022.
  - GraphXAI, Agarwal et al. Sci Data 2023.
  - GInX-Eval, NeurIPS'23 Workshop (mandatory random baseline).
  - Robust Fidelity, Zheng et al. ICLR 2024 (motivates non-zero mask).
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


# ------------------------------------------------------------------ #
#  primitives
# ------------------------------------------------------------------ #

def _char_score(fp: float, fm: float) -> float:
    """GraphFramEx harmonic: 2*Fp*(1-Fm) / (Fp + 1-Fm). Both inputs in [0, 1]."""
    fp = max(0.0, min(1.0, float(fp)))
    fm = max(0.0, min(1.0, float(fm)))
    num = 2.0 * fp * (1.0 - fm)
    denom = fp + (1.0 - fm) + 1e-8
    return num / denom


def _target_prob(logits: torch.Tensor, y: int) -> float:
    """Probability assigned by model to class y. Supports single-output
    (binary, [B, 1]) and multi-class ([B, C])."""
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.size(-1) == 1:
        p = torch.sigmoid(logits.squeeze(-1))
        return float(p.item()) if y == 1 else float((1.0 - p).item())
    p = torch.softmax(logits, dim=-1)[0, y]
    return float(p.item())


def _atom_saliency(model, data: Data, y: int, device) -> torch.Tensor:
    """L1-norm gradient saliency per atom for a single-graph Data.

    Saliency_a = sum_d | d logit_y / d x_{a,d} |.
    Returns: [n_atoms] tensor on CPU.
    """
    model.eval()
    orig_x = data.x
    x = orig_x.detach().clone().requires_grad_(True)
    data.x = x
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    logits = model(data)
    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)
    if logits.size(-1) == 1:
        # binary single-output: target scalar = logit (for y=1) or -logit (y=0)
        target_scalar = logits.squeeze(-1)[0]
        if y == 0:
            target_scalar = -target_scalar
    else:
        target_scalar = logits[0, y]
    grad = torch.autograd.grad(target_scalar, x, retain_graph=False)[0]
    data.x = orig_x
    return grad.abs().sum(dim=1).detach().cpu()


def _ablate_subgraph(data: Data, rm_idx: torch.Tensor, device) -> Data:
    """Build a new Data with the atoms in rm_idx removed entirely
    (incident edges dropped, atoms removed from pool)."""
    n = data.x.size(0)
    keep_mask = torch.ones(n, dtype=torch.bool)
    keep_mask[rm_idx] = False
    new_x = data.x[keep_mask].to(device)
    new_ei, new_ea = subgraph(
        keep_mask, data.edge_index.cpu(),
        edge_attr=data.edge_attr.cpu() if getattr(data, 'edge_attr', None) is not None else None,
        relabel_nodes=True, num_nodes=n,
    )
    new_data = Data(x=new_x, edge_index=new_ei.to(device))
    if new_ea is not None:
        new_data.edge_attr = new_ea.to(device)
    new_data.batch = torch.zeros(new_x.size(0), dtype=torch.long, device=device)
    new_data.y = data.y
    return new_data


def _ablate_edges(data: Data, rm_edge_idx: torch.Tensor, device) -> Data:
    """Remove specific edges from the graph. Atoms stay in pool."""
    n_edges = data.edge_index.size(1)
    keep = torch.ones(n_edges, dtype=torch.bool)
    keep[rm_edge_idx] = False
    new_data = Data(
        x=data.x.to(device),
        edge_index=data.edge_index[:, keep].to(device),
    )
    if getattr(data, 'edge_attr', None) is not None:
        new_data.edge_attr = data.edge_attr[keep].to(device)
    new_data.batch = data.batch.to(device) if hasattr(data, 'batch') and data.batch is not None \
        else torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    new_data.y = data.y
    return new_data


def _edge_saliency_from_atom(atom_saliency: torch.Tensor,
                             edge_index: torch.Tensor) -> torch.Tensor:
    """Edge saliency = sum of endpoint atom saliencies."""
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    return atom_saliency[src] + atom_saliency[dst]


def _prob_of(model, data: Data, y: int) -> float:
    """Forward + return prob assigned to class y. Returns NaN if graph empty."""
    if data.x.size(0) == 0:
        return float('nan')
    with torch.no_grad():
        logits = model(data)
    return _target_prob(logits, y)


def _topk_count(n_atoms: int, frac: float) -> int:
    """At least 1 atom, at most n_atoms - 1 (leave at least one)."""
    k = max(1, int(round(frac * n_atoms)))
    return min(k, max(1, n_atoms - 1))


# ------------------------------------------------------------------ #
#  per-graph fidelity
# ------------------------------------------------------------------ #

def graph_fidelity_one(
    model,
    data: Data,
    device,
    topk_frac: float = 0.1,
    seed: int = 0,
):
    """Compute per-graph fidelity dict for one mol.

    Caller is responsible for filtering to clean-correct positives;
    this routine does not check y or pred itself, but reports
    clean_prob so the caller can decide skip.

    Returns dict with keys: clean_prob, n_atoms, k,
      fp_sal, fm_sal, fp_rand, fm_rand, char_sal, char_rand, delta_char.
    NaN if graph too small.
    """
    n = data.x.size(0)
    if n < 4:
        return {'n_atoms': n, 'k': 0, 'clean_prob': float('nan'),
                'fp_sal': float('nan'), 'fm_sal': float('nan'),
                'fp_rand': float('nan'), 'fm_rand': float('nan'),
                'char_sal': float('nan'), 'char_rand': float('nan'),
                'delta_char': float('nan')}

    y = int(data.y.item()) if data.y.numel() == 1 else int(data.y.view(-1)[0].item())
    data = data.to(device)
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(n, dtype=torch.long, device=device)

    # 1) clean prob
    clean_prob = _prob_of(model, data, y)

    # 2) saliency (needs grad on x)
    sal = _atom_saliency(model, data, y, device)  # [n], cpu

    # 3) pick atom sets
    k = _topk_count(n, topk_frac)
    sal_order = torch.argsort(sal, descending=True)
    top_idx = sal_order[:k]
    bot_idx = sal_order[-k:]

    g = torch.Generator(device='cpu').manual_seed(seed)
    rand_top_idx = torch.randperm(n, generator=g)[:k]
    rand_bot_idx = torch.randperm(n, generator=g)[:k]

    # 4) ablate + re-forward
    p_top_sal = _prob_of(model, _ablate_subgraph(data, top_idx, device), y)
    p_bot_sal = _prob_of(model, _ablate_subgraph(data, bot_idx, device), y)
    p_top_rand = _prob_of(model, _ablate_subgraph(data, rand_top_idx, device), y)
    p_bot_rand = _prob_of(model, _ablate_subgraph(data, rand_bot_idx, device), y)

    fp_sal = clean_prob - p_top_sal
    fm_sal = clean_prob - p_bot_sal
    fp_rand = clean_prob - p_top_rand
    fm_rand = clean_prob - p_bot_rand

    char_sal = _char_score(fp_sal, fm_sal)
    char_rand = _char_score(fp_rand, fm_rand)
    delta = char_sal - char_rand

    return {
        'n_atoms': n, 'k': k, 'clean_prob': clean_prob,
        'fp_sal': fp_sal, 'fm_sal': fm_sal,
        'fp_rand': fp_rand, 'fm_rand': fm_rand,
        'char_sal': char_sal, 'char_rand': char_rand,
        'delta_char': delta,
        'saliency': sal,  # for downstream motif eval
    }


# ------------------------------------------------------------------ #
#  test-set sweep
# ------------------------------------------------------------------ #

def run_graph_fidelity(
    model,
    test_graphs,
    device,
    topk_fracs=(0.1, 0.2),
    seed: int = 0,
    target_label: int = 1,
    motif_masks: dict | None = None,
):
    """Iterate test mols, keep clean-correct positives, compute fidelity
    at each topk_frac, optionally compute motif metrics if mask provided.

    Args:
      test_graphs: iterable of PyG Data (single-graph each)
      target_label: only eval mols with y == this AND pred == this
      motif_masks: optional {graph_idx: bool tensor[n_atoms]} for
        plausibility evaluation (e.g. MUTAG NO2/NH2).

    Returns: dict with per-frac aggregates + n_eval + per-mol records.
    """
    model.eval()
    records = []  # list of dicts, one per kept mol
    n_total = 0
    n_pos = 0
    n_correct = 0  # clean-correct positives
    n_eval = 0    # actually used for fidelity (ideally == n_correct)

    # First pass: collect all positives, separate clean-correct from rest
    pos_correct = []  # list of (gi, data)
    pos_all = []
    for gi, data in enumerate(test_graphs):
        n_total += 1
        if not hasattr(data, 'y') or data.y is None:
            continue
        y = int(data.y.view(-1)[0].item())
        if y != target_label:
            continue
        n_pos += 1
        pos_all.append((gi, data))
        data_dev = data.to(device)
        if not hasattr(data_dev, 'batch') or data_dev.batch is None:
            data_dev.batch = torch.zeros(data_dev.x.size(0), dtype=torch.long,
                                         device=device)
        with torch.no_grad():
            logits = model(data_dev)
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)
        if logits.size(-1) == 1:
            pred = int((torch.sigmoid(logits.squeeze(-1)).item() >= 0.5))
        else:
            pred = int(logits.argmax(-1).item())
        if pred == target_label:
            n_correct += 1
            pos_correct.append((gi, data))

    # Eval set: prefer clean-correct positives; fallback to all positives
    # if too few correct (model under-trained → smoke / weak FT cell)
    used_fallback = False
    if n_correct >= 10:
        eval_set = pos_correct
    else:
        eval_set = pos_all
        used_fallback = True
        print(f"[GINTERP] n_correct={n_correct} < 10, falling back to all "
              f"{len(pos_all)} y=1 mols (eval less precise on this cell)")

    for gi, data in eval_set:
        n_eval += 1

        per_mol = {'graph_idx': gi}
        sal_cache = None
        for frac in topk_fracs:
            r = graph_fidelity_one(model, data, device,
                                   topk_frac=frac, seed=seed + gi)
            for k, v in r.items():
                if k == 'saliency':
                    sal_cache = v
                    continue
                per_mol[f'{k}@{int(frac*100)}'] = v

        # motif metrics (k=10% by convention)
        if motif_masks is not None and gi in motif_masks and sal_cache is not None:
            gt = motif_masks[gi]
            if gt.numel() == sal_cache.numel() and gt.sum() > 0:
                per_mol['motif_prec@10'] = _motif_precision(sal_cache, gt, 0.1)
                per_mol['motif_auc'] = _motif_auc(sal_cache, gt)
                # random baseline = motif size / total atoms
                per_mol['motif_prec_rand'] = float(gt.sum().item()) / float(gt.numel())

        records.append(per_mol)

    # aggregate
    agg = {'n_total': n_total, 'n_pos': n_pos,
           'n_correct': n_correct, 'n_eval': n_eval,
           'fallback_used': int(used_fallback)}
    if records:
        keys = [k for k in records[0].keys() if k != 'graph_idx']
        for k in keys:
            vs = [r[k] for r in records
                  if k in r and r[k] is not None
                  and not (isinstance(r[k], float) and np.isnan(r[k]))]
            if vs:
                agg[f'{k}_mean'] = float(np.mean(vs))
                agg[f'{k}_std'] = float(np.std(vs))
            else:
                agg[f'{k}_mean'] = float('nan')
                agg[f'{k}_std'] = float('nan')
    return agg, records


# ------------------------------------------------------------------ #
#  motif metrics (used when ground-truth motif mask available)
# ------------------------------------------------------------------ #

def _motif_precision(saliency: torch.Tensor, motif_mask: torch.Tensor,
                     frac: float = 0.1) -> float:
    n = saliency.numel()
    k = max(1, int(round(frac * n)))
    top_idx = torch.argsort(saliency, descending=True)[:k]
    return float(motif_mask[top_idx].float().mean().item())


def _motif_auc(saliency: torch.Tensor, motif_mask: torch.Tensor) -> float:
    """Simple AUROC: rank-based. Returns 0.5 if degenerate."""
    sal = saliency.numpy().astype(float)
    gt = motif_mask.numpy().astype(int)
    if gt.sum() == 0 or gt.sum() == gt.size:
        return float('nan')
    order = np.argsort(-sal)
    gt_sorted = gt[order]
    n_pos = gt_sorted.sum()
    n_neg = len(gt_sorted) - n_pos
    # rank-sum AUROC (avg rank of positives)
    ranks = np.arange(1, len(gt_sorted) + 1)
    sum_pos_ranks = ranks[gt_sorted == 1].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


# ------------------------------------------------------------------ #
#  log formatters
# ------------------------------------------------------------------ #

def format_raw_log(method: str, dataset: str, split: str, seed: int,
                   agg: dict) -> str:
    parts = [
        f"method={method}", f"dataset={dataset}", f"split={split}", f"seed={seed}",
        f"n_total={agg.get('n_total', 0)}", f"n_pos={agg.get('n_pos', 0)}",
        f"n_eval={agg.get('n_eval', 0)}",
    ]
    for k in sorted(agg.keys()):
        if k in ('n_total', 'n_pos', 'n_eval'):
            continue
        v = agg[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "[GINTERP_V2_RAW] " + " ".join(parts)
