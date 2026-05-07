"""LLaGA generative FT + OOD (degree-shift) evaluation.

Uses AnyGraph's degree-split scheme: sort labeled nodes by degree,
take top 60% as train_pool (random 10/10 carved out as id_val/id_test),
middle 20% as ood_val, bottom 20% as ood_test. Report acc+f1 on both
id_test and ood_test with the answer-ranking generative inference.
"""

from __future__ import annotations

import argparse
import collections
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag
from sgb.models.gfm.llaga.label_names import get_label_names
from sgb.models.gfm.llaga.model import (
    LlagaConfig,
    build_nd_subgraph_indices,
    lookup_nd_features,
)
from sgb.models.gfm.llaga.run_node import (
    LlagaGenerative,
    _get_feat_768d,
    _build_prompt_only_ids,
    _pack_text_batch,
    _rank_answers,
)
# ---------------------------------------------------------------------------
# Self-contained degree-OOD split (GOOD 60/20/20 covariate shift).
# Identical protocol to other benchmark methods but implemented here so LLaGA
# has zero cross-method imports.
# ---------------------------------------------------------------------------


def _compute_node_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    deg = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(edge_index.size(1), dtype=torch.long)
    deg.scatter_add_(0, edge_index[0].cpu().long(), ones)
    deg.scatter_add_(0, edge_index[1].cpu().long(), ones)
    return deg


def build_degree_split(dataset_name, edge_index, labels, split_seed):
    """Sort labeled nodes by degree desc; top 60% = train_pool (with 10/10
    carved out for id_val/id_test); middle 20% = ood_val; bottom 20% = ood_test.
    """
    labels_cpu = labels.detach().cpu().long()
    num_nodes = int(labels_cpu.numel())
    labeled_bool = labels_cpu >= 0
    labeled_idx_all = torch.arange(num_nodes)[labeled_bool]
    labeled_y_all = labels_cpu[labeled_idx_all]
    deg = _compute_node_degree(edge_index, num_nodes)
    labeled_deg = deg[labeled_idx_all].long()
    num_classes_total = int(torch.unique(labeled_y_all).numel())

    sort_key = labeled_deg * (num_nodes + 1) + labeled_idx_all.long()
    order = torch.argsort(sort_key, descending=True)
    sorted_idx = labeled_idx_all[order]

    n_labeled = int(sorted_idx.numel())
    train_end = int(round(n_labeled * 0.60))
    ood_val_end = int(round(n_labeled * 0.80))
    train_pool_idx = sorted_idx[:train_end]
    ood_val_idx = sorted_idx[train_end:ood_val_end]
    ood_test_idx = sorted_idx[ood_val_end:]

    num_id = int(round(n_labeled * 0.10))
    if 2 * num_id >= train_pool_idx.numel() or train_pool_idx.numel() == 0:
        actual_train_idx = train_pool_idx
        id_val_idx = torch.empty(0, dtype=torch.long)
        id_test_idx = torch.empty(0, dtype=torch.long)
    else:
        rng = np.random.RandomState(split_seed)
        perm = torch.as_tensor(rng.permutation(int(train_pool_idx.numel())),
                               dtype=torch.long)
        shuffled = train_pool_idx[perm]
        actual_train_idx = shuffled[: -2 * num_id]
        id_val_idx = shuffled[-2 * num_id : -num_id]
        id_test_idx = shuffled[-num_id :]

    return {
        "train": actual_train_idx, "id_val": id_val_idx, "id_test": id_test_idx,
        "ood_val": ood_val_idx, "ood_test": ood_test_idx,
        "meta": {
            "dataset": dataset_name, "split_seed": split_seed, "shift": "degree",
            "strategy": "good_60_20_20_descending",
            "num_classes": int(num_classes_total), "num_nodes_total": int(num_nodes),
            "n_labeled": int(n_labeled),
            "train_pool_size": int(train_pool_idx.numel()),
            "actual_train_size": int(actual_train_idx.numel()),
            "id_val_size": int(id_val_idx.numel()), "id_test_size": int(id_test_idx.numel()),
            "ood_val_size": int(ood_val_idx.numel()), "ood_test_size": int(ood_test_idx.numel()),
        },
    }


def _idx_to_mask(idx, N, device):
    m = torch.zeros(N, dtype=torch.bool, device=device)
    if idx.numel() > 0:
        m[idx.to(device)] = True
    return m


@torch.no_grad()
def eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y, idx, label_names,
                llm_dtype, eval_batch_size=16):
    if idx.numel() == 0:
        return float("nan"), float("nan")
    model.eval()
    device = next(model.parameters()).device
    preds, trues = [], []
    for start in range(0, idx.numel(), eval_batch_size):
        end = min(start + eval_batch_size, idx.numel())
        batch = idx[start:end]
        nd_batch = nd_indices[batch].to(device)
        tok, mask = lookup_nd_features(nd_batch, feat)
        best_cls = _rank_answers(model, tokenizer, prompt_ids, tok, mask,
                                 label_names, llm_dtype)
        preds.append(best_cls)
        trues.append(y[batch].cpu())
    preds = torch.cat(preds, dim=0); trues = torch.cat(trues, dim=0)
    acc = (preds == trues).float().mean().item() * 100.0
    f1 = f1_score(trues.numpy(), preds.numpy(), average="macro") * 100.0
    return acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--llm_name", default="lmsys/vicuna-7b-v1.5-16k")
    p.add_argument("--llm_dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="eager")
    p.add_argument("--projector_type", default="2-layer-mlp")
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--use_hop", default=2, type=int)
    p.add_argument("--sample_size", default=10, type=int)
    p.add_argument("--max_epochs", default=15, type=int)
    p.add_argument("--patience", default=3, type=int)
    p.add_argument("--val_every", default=3, type=int)
    p.add_argument("--val_subsample", default=128, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--batch_size", default=8, type=int)
    p.add_argument("--eval_batch_size", default=16, type=int)
    p.add_argument("--num_splits", default=5, type=int)
    p.add_argument("--test_subsample", default=1000, type=int)
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-OOD] device={device} llm={args.llm_name}")

    data, _ = load_tag(args.dataset)
    feat = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long() if data.y.dim() > 1 else data.y.long()
    y = y.to(device)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data)
    if not label_names:
        raise RuntimeError(f"no label_names for {args.dataset}")
    num_classes = len(label_names)
    print(f"[LLAGA-OOD] {args.dataset}: N={N} C={num_classes}")

    cfg = LlagaConfig(
        llm_name_or_path=args.llm_name, mm_hidden_size=feat.size(1),
        projector_type=args.projector_type,
        use_hop=args.use_hop, sample_size=args.sample_size,
        freeze_llm=True, llm_dtype=args.llm_dtype,
        cache_dir=args.cache_dir, attn_implementation=args.attn_impl,
    )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, cache_dir=cfg.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_text = (
        f"Task: {args.dataset} node classification. "
        f"Given the graph context, predict the class. Answer: "
    )
    prompt_ids = _build_prompt_only_ids(tokenizer, prompt_text)

    all_results = []
    for split_idx in range(args.num_splits):
        torch.manual_seed(split_idx); np.random.seed(split_idx)

        # Build degree split (on cpu)
        degsplit = build_degree_split(args.dataset, edge_index, y.cpu(), split_seed=split_idx)
        train_idx_cpu = degsplit["train"]
        id_val_cpu = degsplit["id_val"]
        id_test_cpu = degsplit["id_test"]
        ood_val_cpu = degsplit["ood_val"]
        ood_test_cpu = degsplit["ood_test"]
        meta = degsplit["meta"]
        print(f"[LLAGA-OOD] split={split_idx} train={len(train_idx_cpu)} "
              f"id_val={len(id_val_cpu)} id_test={len(id_test_cpu)} "
              f"ood_val={len(ood_val_cpu)} ood_test={len(ood_test_cpu)}", flush=True)

        # Optionally cap eval sets
        def _cap(idx, cap, seed):
            if cap <= 0 or idx.numel() <= cap:
                return idx
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(idx.numel(), generator=g)[:cap]
            return idx[perm]
        id_test_eval = _cap(id_test_cpu, args.test_subsample, split_idx * 31 + 1)
        ood_test_eval = _cap(ood_test_cpu, args.test_subsample, split_idx * 31 + 2)

        # ND indices (all N)
        print(f"[LLAGA-OOD] split={split_idx}: building ND...", flush=True)
        nd_indices = build_nd_subgraph_indices(
            edge_index, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size, seed=split_idx,
        )

        model = LlagaGenerative(cfg).to(device)
        if args.projector_ckpt:
            state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
            model.projector.load_state_dict(state, strict=True)
        optim = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=args.wd)
        llm_dtype = next(model.llm.parameters()).dtype

        # Pre-build val subsample for early stopping (use id_val; fall back to train if empty)
        id_val_pool = id_val_cpu if id_val_cpu.numel() > 0 else train_idx_cpu
        if args.val_subsample > 0 and id_val_pool.numel() > args.val_subsample:
            gv = torch.Generator().manual_seed(split_idx * 17 + 5)
            val_eval = id_val_pool[torch.randperm(id_val_pool.numel(), generator=gv)[:args.val_subsample]]
        else:
            val_eval = id_val_pool

        # Training loop
        best_val = -1.0; best_state = None; no_improve = 0
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            perm = torch.randperm(train_idx_cpu.numel())
            train_perm = train_idx_cpu[perm]
            total_loss = 0.0; n_b = 0
            for start in range(0, train_perm.numel(), args.batch_size):
                end = min(start + args.batch_size, train_perm.numel())
                batch = train_perm[start:end]
                nd_batch = nd_indices[batch].to(device)
                tok, mask = lookup_nd_features(nd_batch, feat)
                ans = [label_names[int(y[batch[b]].item())] for b in range(batch.numel())]
                prs = [prompt_text] * batch.numel()
                inp_ids, tgts, tmask = _pack_text_batch(tokenizer, prs, ans, args.max_text_len)
                inp_ids = inp_ids.to(device); tgts = tgts.to(device); tmask = tmask.to(device)
                loss = model.forward_loss(tok, mask, inp_ids, tmask, tgts, llm_dtype)
                optim.zero_grad(); loss.backward(); optim.step()
                total_loss += float(loss.item()); n_b += 1

            if epoch % args.val_every == 0 or epoch == args.max_epochs:
                val_acc, _ = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                         val_eval, label_names, llm_dtype, args.eval_batch_size)
                avg = total_loss / max(n_b, 1)
                print(f"[OOD-GEN] split={split_idx} epoch={epoch} train_loss={avg:.4f} id_val={val_acc:.4f}", flush=True)
                if val_acc > best_val:
                    best_val = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        break

        if best_state is not None:
            model.projector.load_state_dict(best_state)

        # Final eval on id_test and ood_test
        id_acc, id_f1 = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                    id_test_eval, label_names, llm_dtype, args.eval_batch_size)
        ood_acc, ood_f1 = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                      ood_test_eval, label_names, llm_dtype, args.eval_batch_size)
        gap = id_acc - ood_acc if id_acc == id_acc and ood_acc == ood_acc else float("nan")
        print(f"[OOD_RAW] method=LLaGA_GEN dataset={args.dataset} split_idx={split_idx} "
              f"seed={split_idx} id_test_acc={id_acc:.4f} id_test_f1={id_f1:.4f} "
              f"ood_test_acc={ood_acc:.4f} ood_test_f1={ood_f1:.4f} gap={gap:.4f}", flush=True)
        all_results.append({"split": split_idx, "id_acc": id_acc, "id_f1": id_f1,
                            "ood_acc": ood_acc, "ood_f1": ood_f1, "gap": gap})

        del model; torch.cuda.empty_cache()

    # Aggregate
    print(f"\n=== LLaGA GEN OOD ({args.dataset}) ===")
    def _agg(k):
        arr = np.array([r[k] for r in all_results])
        return f"{np.nanmean(arr):.2f} ± {np.nanstd(arr):.2f}"
    print(f"[OOD_AGG] method=LLaGA_GEN dataset={args.dataset} "
          f"id_test_acc=\"{_agg('id_acc')}\" id_test_f1=\"{_agg('id_f1')}\" "
          f"ood_test_acc=\"{_agg('ood_acc')}\" ood_test_f1=\"{_agg('ood_f1')}\" "
          f"gap=\"{_agg('gap')}\"")


if __name__ == "__main__":
    main()
