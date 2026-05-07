"""LLaGA generative FT + feature-noise OR edge-deletion eval (5 severities each).

Trains clean (same as run_node), then runs answer-ranking eval at
each severity. Saves [FN_RAW]/[FN_AGG] or [ED_RAW]/[ED_AGG] lines.
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
    _get_splits,
    _build_prompt_only_ids,
    train_and_eval,
    _rank_answers,
)
def apply_edge_drop(edge_index, num_nodes, p, seed):
    """Bernoulli edge drop aligned with BGRL/GraphMAE/GIT/GCOPE/UniGraph2 convention."""
    if p <= 0.0 or edge_index.size(1) == 0:
        return edge_index
    g = torch.Generator(device=edge_index.device).manual_seed(int(seed))
    src, dst = edge_index[0], edge_index[1]
    u = torch.minimum(src, dst)
    v = torch.maximum(src, dst)
    key = u.long() * num_nodes + v.long()
    _, inverse = torch.unique(key, return_inverse=True)
    num_undirected = int(inverse.max().item()) + 1
    keep = (torch.rand(num_undirected, generator=g, device=edge_index.device) >= p)[inverse]
    keep = keep | (src == dst)
    return edge_index[:, keep]


FN_SEVERITIES = [(1, 0.1), (2, 0.25), (3, 0.5), (4, 1.0), (5, 2.0)]
# Same ED severities as BGRL/GraphMAE/GIT/GCOPE/UniGraph2/gnn_baseline
ED_SEVERITIES = [(1, 0.05), (2, 0.10), (3, 0.20), (4, 0.30), (5, 0.50)]


def apply_feature_noise(x, train_mask, sigma_rel, noise_seed):
    if train_mask.dtype != torch.bool:
        train_mask = train_mask.bool()
    std = x[train_mask].std(dim=0, keepdim=True)
    g = torch.Generator(device=x.device).manual_seed(int(noise_seed))
    eps = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


@torch.no_grad()
def eval_ranked(model, tokenizer, prompt_ids, feat_eval, nd_indices, y, test_idx,
                label_names, llm_dtype, eval_batch_size=16):
    """Run answer-ranking over test_idx, returning (acc, macro_f1)."""
    model.eval()
    device = next(model.parameters()).device
    preds, trues = [], []
    for start in range(0, test_idx.numel(), eval_batch_size):
        end = min(start + eval_batch_size, test_idx.numel())
        batch_nodes = test_idx[start:end]
        nd_batch = nd_indices[batch_nodes].to(device)
        tok, mask = lookup_nd_features(nd_batch, feat_eval)
        best_cls = _rank_answers(model, tokenizer, prompt_ids, tok, mask,
                                 label_names, llm_dtype)
        preds.append(best_cls)
        trues.append(y[batch_nodes].cpu())
    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    acc = (preds == trues).float().mean().item() * 100.0
    f1 = f1_score(trues.numpy(), preds.numpy(), average="macro") * 100.0
    return acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corruption", required=True, choices=["feature_noise", "edge_deletion"])
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
    p.add_argument("--test_subsample", default=1000, type=int,
                   help="Cap test-set size to speed up eval; -1 = use full")
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    is_fn = args.corruption == "feature_noise"
    sevs = FN_SEVERITIES if is_fn else ED_SEVERITIES
    tag = "FN" if is_fn else "ED"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-{tag}] device={device} llm={args.llm_name}")

    data, _ = load_tag(args.dataset)
    feat_clean = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long().to(device) if data.y.dim() > 1 else data.y.long().to(device)
    N = feat_clean.size(0)
    edge_index_clean = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data)
    if not label_names:
        raise RuntimeError(f"no label_names for {args.dataset}")
    splits = _get_splits(data, N)
    print(f"[LLAGA-{tag}] {args.dataset}: N={N} classes={len(label_names)} test_sub={args.test_subsample}")

    cfg = LlagaConfig(
        llm_name_or_path=args.llm_name, mm_hidden_size=feat_clean.size(1),
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
    for split_idx, split in enumerate(splits[: args.num_splits]):
        torch.manual_seed(split_idx); np.random.seed(split_idx)

        print(f"[LLAGA-{tag}] split={split_idx}: clean ND indices...", flush=True)
        nd_clean = build_nd_subgraph_indices(
            edge_index_clean, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size, seed=split_idx,
        )

        model = LlagaGenerative(cfg).to(device)
        if args.projector_ckpt:
            state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
            model.projector.load_state_dict(state, strict=True)

        train_mask = split["train"].to(device)
        val_mask = split["val"].to(device)
        test_mask = split["test"].to(device)
        test_idx_full = torch.nonzero(test_mask, as_tuple=False).squeeze(-1).cpu()
        # Cap test set for speed
        if args.test_subsample > 0 and test_idx_full.numel() > args.test_subsample:
            g = torch.Generator().manual_seed(split_idx * 9973 + 1)
            perm = torch.randperm(test_idx_full.numel(), generator=g)[:args.test_subsample]
            test_idx = test_idx_full[perm]
        else:
            test_idx = test_idx_full
        llm_dtype = next(model.llm.parameters()).dtype

        # Clean FT + clean acc reported via train_and_eval's own final test. We
        # reuse it but ignore its returned number: we re-run eval ourselves on
        # the subsampled test_idx so sev=0 and sevs>0 share the same test set.
        train_and_eval(
            model, tokenizer, feat_clean, nd_clean, y,
            train_mask, val_mask, test_mask, label_names, prompt_text,
            device, llm_dtype,
            max_epochs=args.max_epochs, patience=args.patience,
            lr=args.lr, wd=args.wd,
            batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            max_text_len=args.max_text_len,
            val_every=args.val_every, val_subsample=args.val_subsample,
        )

        # clean eval on test_idx
        clean_acc, clean_f1 = eval_ranked(
            model, tokenizer, prompt_ids, feat_clean, nd_clean, y, test_idx,
            label_names, llm_dtype, args.eval_batch_size,
        )
        print(f"[{tag}_RAW] method=LLaGA_GEN dataset={args.dataset} split_idx={split_idx} "
              f"seed={split_idx} sev=0 {'sigma_rel' if is_fn else 'drop_rate'}=0.0 "
              f"test_acc={clean_acc:.4f} macro_f1={clean_f1:.4f}", flush=True)
        all_results.append({"split": split_idx, "sev": 0, "acc": clean_acc, "f1": clean_f1})

        for sev_idx, param in sevs:
            if is_fn:
                feat_pert = apply_feature_noise(
                    feat_clean, train_mask, param,
                    noise_seed=split_idx * 100 + sev_idx,
                )
                acc, f1 = eval_ranked(
                    model, tokenizer, prompt_ids, feat_pert, nd_clean, y, test_idx,
                    label_names, llm_dtype, args.eval_batch_size,
                )
                kvs = f"sigma_rel={param}"
            else:
                ei_pert = apply_edge_drop(
                    edge_index_clean, N, param, seed=split_idx * 100 + sev_idx,
                )
                # Only rebuild ND for test nodes (massive speedup for large N).
                nd_test_rows = build_nd_subgraph_indices(
                    ei_pert, N, test_idx,
                    use_hop=args.use_hop, sample_size=args.sample_size,
                    seed=split_idx * 100 + sev_idx,
                )  # [len(test_idx), L]
                # Scatter into a full-N tensor so eval_ranked's indexing works.
                nd_pert = nd_clean.clone()
                nd_pert[test_idx] = nd_test_rows
                acc, f1 = eval_ranked(
                    model, tokenizer, prompt_ids, feat_clean, nd_pert, y, test_idx,
                    label_names, llm_dtype, args.eval_batch_size,
                )
                kvs = f"drop_rate={param}"

            print(f"[{tag}_RAW] method=LLaGA_GEN dataset={args.dataset} split_idx={split_idx} "
                  f"seed={split_idx} sev={sev_idx} {kvs} "
                  f"test_acc={acc:.4f} macro_f1={f1:.4f}", flush=True)
            all_results.append({"split": split_idx, "sev": sev_idx, "acc": acc, "f1": f1})

        del model
        torch.cuda.empty_cache()

    # Aggregate
    print(f"\n=== LLaGA GEN {tag} Results ({args.dataset}) ===")
    ga = collections.defaultdict(list); gf = collections.defaultdict(list)
    for r in all_results:
        ga[r["sev"]].append(r["acc"]); gf[r["sev"]].append(r["f1"])
    labels = ["clean", "sev1", "sev2", "sev3", "sev4", "sev5"]
    parts_a, parts_f = [], []
    for sev in sorted(ga.keys()):
        a = np.array(ga[sev]); f = np.array(gf[sev])
        parts_a.append(f'{labels[sev]}="{a.mean():.2f} ± {a.std():.2f}"')
        parts_f.append(f'{labels[sev]}="{f.mean():.2f} ± {f.std():.2f}"')
    print(f"[{tag}_AGG] method=LLaGA_GEN dataset={args.dataset} metric=acc " + " ".join(parts_a))
    print(f"[{tag}_AGG] method=LLaGA_GEN dataset={args.dataset} metric=f1 " + " ".join(parts_f))


if __name__ == "__main__":
    main()
