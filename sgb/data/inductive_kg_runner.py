"""Shared inductive KG link-prediction runner.

For FB15K237 / WN18RR with inductive splits built by `inductive_kg_split.py`:
  1. Train encoder + DistMult decoder on G_tr with head/tail corruption negatives.
  2. Every eval_every epochs, evaluate filtered MRR on Q_te_valid (inductive
     validation: encoder sees G_te_support only). Track best for early stop.
  3. At the end, restore best checkpoint and report final filtered
     MRR / Hits@10 on Q_te_test.

Per-method `run_inductive_kg.py` provides a `build_ft_model(in_channels,
device)` factory returning an `nn.Module` exposing `.encode(x, edge_index)
-> z [N, d]`.

Output:
  [INDKG_RAW] method=... dataset=... seed=... split={valid|test}
              mrr=... hits10=... n_query=...
  [INDKG_AGG] method=... dataset=... mrr_mean=... mrr_std=...
              hits10_mean=... hits10_std=... n_seeds=...
"""
from __future__ import annotations

import copy
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))


def _remap_to_compact(edge_index: torch.Tensor, entity_set: torch.Tensor,
                      num_nodes_full: int) -> torch.Tensor:
    """Map edge_index entries (in [0, num_nodes_full)) to compact [0, |entity_set|)
    using entity_set as the canonical ordering. Edges referencing entities
    not in entity_set are dropped (caller should pre-filter)."""
    remap = torch.full((num_nodes_full,), -1, dtype=torch.long)
    remap[entity_set] = torch.arange(len(entity_set))
    new = remap[edge_index]
    assert (new >= 0).all(), "edge references entity outside the set"
    return new


def _build_filter_dict(query_edges: torch.Tensor, query_rels: torch.Tensor,
                       support_edges: torch.Tensor = None,
                       support_rels: torch.Tensor = None):
    """For each (h, r) build set of true tails, and for each (r, t) the true heads."""
    tails_for_hr = {}
    heads_for_rt = {}
    for src in [(query_edges, query_rels), (support_edges, support_rels)]:
        if src[0] is None:
            continue
        ei, et = src
        for k in range(ei.size(1)):
            h, t = int(ei[0, k]), int(ei[1, k])
            r = int(et[k])
            tails_for_hr.setdefault((h, r), set()).add(t)
            heads_for_rt.setdefault((r, t), set()).add(h)
    return tails_for_hr, heads_for_rt


@torch.no_grad()
def _filtered_mrr_hits(model, decoder, x, edge_index, query_edges, query_rels,
                       num_entities, tails_for_hr, heads_for_rt,
                       chunk_query=128, chunk_cand=2048, device="cuda"):
    """Compute filtered MRR / Hits@10 averaged over head and tail prediction."""
    model.eval()
    decoder.eval()
    z = model.encode(x.to(device), edge_index.to(device))  # [num_entities, d]
    rel_emb = decoder.weight  # [num_relations, d]

    n_query = query_edges.size(1)
    rranks_tail = []
    rranks_head = []
    hits_tail = 0
    hits_head = 0

    qe = query_edges.to(device)
    qr = query_rels.to(device)

    for q_start in range(0, n_query, chunk_query):
        q_end = min(q_start + chunk_query, n_query)
        h_idx = qe[0, q_start:q_end]   # [B]
        t_idx = qe[1, q_start:q_end]
        r_idx = qr[q_start:q_end]
        B = h_idx.size(0)

        # ---- Tail prediction: score (h, r, t') for all t'
        zh = z[h_idx]              # [B, d]
        wr = rel_emb[r_idx]        # [B, d]
        hr = zh * wr               # [B, d]
        # score against all candidates in chunks
        scores_tail = torch.empty(B, num_entities, device=device)
        for c_start in range(0, num_entities, chunk_cand):
            c_end = min(c_start + chunk_cand, num_entities)
            zt_chunk = z[c_start:c_end]                      # [C, d]
            scores_tail[:, c_start:c_end] = hr @ zt_chunk.T  # [B, C]

        # ---- Head prediction: score (h', r, t) for all h'
        zt = z[t_idx]
        rt = wr * zt
        scores_head = torch.empty(B, num_entities, device=device)
        for c_start in range(0, num_entities, chunk_cand):
            c_end = min(c_start + chunk_cand, num_entities)
            zh_chunk = z[c_start:c_end]
            scores_head[:, c_start:c_end] = rt @ zh_chunk.T

        # ---- Filtering: mask out other true triples
        for b in range(B):
            h, r, t = int(h_idx[b]), int(r_idx[b]), int(t_idx[b])
            true_tails = tails_for_hr.get((h, r), set())
            for tt in true_tails:
                if tt != t:
                    scores_tail[b, tt] = float("-inf")
            true_heads = heads_for_rt.get((r, t), set())
            for hh in true_heads:
                if hh != h:
                    scores_head[b, hh] = float("-inf")

        # ranks (1-indexed)
        for b in range(B):
            h, t = int(h_idx[b]), int(t_idx[b])
            r_t = int((scores_tail[b] > scores_tail[b, t]).sum().item()) + 1
            r_h = int((scores_head[b] > scores_head[b, h]).sum().item()) + 1
            rranks_tail.append(1.0 / r_t)
            rranks_head.append(1.0 / r_h)
            hits_tail += int(r_t <= 10)
            hits_head += int(r_h <= 10)

    rranks = rranks_tail + rranks_head
    mrr = float(np.mean(rranks))
    hits10 = (hits_tail + hits_head) / (2 * n_query)
    return mrr, hits10


def run_inductive_kg(method_tag: str, dataset: str,
                    build_ft_model: Callable, hidden_dim: int = 768,
                    lr: float = 5e-4, weight_decay: float = 1e-5,
                    max_epochs: int = 100, eval_every: int = 5,
                    patience: int = 5, n_negatives: int = 10,
                    n_seeds: int = 3, pos_batch_size: Optional[int] = None):
    from sgb.data.inductive_kg_split import load_inductive_split
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INDKG] method={method_tag} dataset={dataset} device={device}")

    valid_mrrs, valid_hits, test_mrrs, test_hits = [], [], [], []
    for seed in range(n_seeds):
        sp = load_inductive_split(dataset, partition_seed=seed)
        s = sp["stats"]
        print(f"[INDKG seed={seed}] |E_tr|={s['n_E_tr']} |E_te|={s['n_E_te']} "
              f"|G_tr|={s['n_G_tr']} |G_te_sup|={s['n_G_te_support']} "
              f"|Q_v|={s['n_Q_te_valid']} |Q_t|={s['n_Q_te_test']} "
              f"rels={s['n_relations_in_tr']}/{s['n_relations_total']}")

        # Build train graph (compact ids on E_tr)
        E_tr = sp["E_tr"]
        E_te = sp["E_te"]
        N = sp["num_nodes_full"]
        feat = sp["node_text_feat"]

        tr_x = feat[E_tr]                                   # [|E_tr|, d]
        tr_edge_index = _remap_to_compact(sp["G_tr_edge_index"], E_tr, N)
        tr_rels = sp["G_tr_edge_types"]

        te_x = feat[E_te]
        te_support_ei = _remap_to_compact(sp["G_te_support_edge_index"], E_te, N)
        te_support_rels = sp["G_te_support_edge_types"]
        te_valid_ei = _remap_to_compact(sp["Q_te_valid_edge_index"], E_te, N)
        te_valid_rels = sp["Q_te_valid_edge_types"]
        te_test_ei = _remap_to_compact(sp["Q_te_test_edge_index"], E_te, N)
        te_test_rels = sp["Q_te_test_edge_types"]

        num_relations = int(s["n_relations_total"])
        num_E_tr = int(s["n_E_tr"])
        num_E_te = int(s["n_E_te"])

        # Build model + decoder
        torch.manual_seed(seed)
        model = build_ft_model(feat.size(1), device)
        # DistMult decoder: relation embedding [num_relations, hidden_dim]
        decoder = nn.Embedding(num_relations, hidden_dim).to(device)
        nn.init.xavier_uniform_(decoder.weight)

        params = list(model.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # Pre-build filter dicts for valid/test eval
        # Filter against all known triples within G_te (support + valid + test)
        all_te_ei = torch.cat([te_support_ei, te_valid_ei, te_test_ei], dim=1)
        all_te_rels = torch.cat([te_support_rels, te_valid_rels, te_test_rels], dim=0)
        tails_te, heads_te = _build_filter_dict(all_te_ei, all_te_rels)

        tr_x_dev = tr_x.to(device)
        tr_ei_dev = tr_edge_index.to(device)
        tr_rels_dev = tr_rels.to(device)

        best_valid_mrr = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            decoder.train()
            opt.zero_grad()

            z = model.encode(tr_x_dev, tr_ei_dev)  # [num_E_tr, d]

            E = tr_ei_dev.size(1)
            if pos_batch_size is not None and E > pos_batch_size:
                idx = torch.randint(0, E, (pos_batch_size,), device=device)
                ei_step = tr_ei_dev[:, idx]
                rels_step = tr_rels_dev[idx]
            else:
                ei_step = tr_ei_dev
                rels_step = tr_rels_dev
            n_pos = ei_step.size(1)
            # head/tail corruption negatives (within E_tr)
            neg_t = torch.randint(0, num_E_tr, (n_pos, n_negatives), device=device)
            neg_h = torch.randint(0, num_E_tr, (n_pos, n_negatives), device=device)

            zh = z[ei_step[0]]                      # [n_pos, d]
            zt = z[ei_step[1]]
            wr = decoder(rels_step)                 # [n_pos, d]

            pos_score = (zh * wr * zt).sum(-1)      # [E]
            zt_neg = z[neg_t]                       # [E, k, d]
            zh_neg = z[neg_h]
            neg_score_t = (zh.unsqueeze(1) * wr.unsqueeze(1) * zt_neg).sum(-1)  # [E, k]
            neg_score_h = (zh_neg * wr.unsqueeze(1) * zt.unsqueeze(1)).sum(-1)
            neg_score = torch.cat([neg_score_t, neg_score_h], dim=1)            # [E, 2k]

            # Cross-entropy with negatives (Bordes-style softplus)
            pos_loss = F.softplus(-pos_score).mean()
            neg_loss = F.softplus(neg_score).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            opt.step()

            if epoch % eval_every == 0 or epoch == max_epochs:
                # Inductive validation: encode te_support, eval Q_te_valid
                v_mrr, v_h10 = _filtered_mrr_hits(
                    model, decoder, te_x, te_support_ei,
                    te_valid_ei, te_valid_rels,
                    num_entities=num_E_te,
                    tails_for_hr=tails_te, heads_for_rt=heads_te,
                    device=device,
                )
                print(f"[INDKG seed={seed}] epoch={epoch} loss={loss.item():.4f} "
                      f"valid_mrr={v_mrr:.4f} valid_hits10={v_h10:.4f}")
                if v_mrr > best_valid_mrr + 1e-4:
                    best_valid_mrr = v_mrr
                    best_state = {
                        "model": copy.deepcopy(model.state_dict()),
                        "decoder": copy.deepcopy(decoder.state_dict()),
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"[INDKG seed={seed}] early stop at epoch {epoch}")
                    break

        # Restore best, evaluate on Q_te_test
        if best_state is not None:
            model.load_state_dict(best_state["model"])
            decoder.load_state_dict(best_state["decoder"])
        v_mrr, v_h10 = _filtered_mrr_hits(
            model, decoder, te_x, te_support_ei,
            te_valid_ei, te_valid_rels,
            num_entities=num_E_te,
            tails_for_hr=tails_te, heads_for_rt=heads_te, device=device)
        t_mrr, t_h10 = _filtered_mrr_hits(
            model, decoder, te_x, te_support_ei,
            te_test_ei, te_test_rels,
            num_entities=num_E_te,
            tails_for_hr=tails_te, heads_for_rt=heads_te, device=device)
        valid_mrrs.append(v_mrr); valid_hits.append(v_h10)
        test_mrrs.append(t_mrr); test_hits.append(t_h10)
        print(f"[INDKG_RAW] method={method_tag} dataset={dataset} seed={seed} "
              f"split=valid mrr={v_mrr:.4f} hits10={v_h10:.4f} n_query={te_valid_ei.size(1)}")
        print(f"[INDKG_RAW] method={method_tag} dataset={dataset} seed={seed} "
              f"split=test mrr={t_mrr:.4f} hits10={t_h10:.4f} n_query={te_test_ei.size(1)}")

    if test_mrrs:
        print(f"[INDKG_AGG] method={method_tag} dataset={dataset} "
              f"mrr_mean={np.mean(test_mrrs):.4f} mrr_std={np.std(test_mrrs):.4f} "
              f"hits10_mean={np.mean(test_hits):.4f} hits10_std={np.std(test_hits):.4f} "
              f"n_seeds={len(test_mrrs)}")
