"""LLaGA generative FT + step-imbalance evaluation.

Applies TAM/ReNode step-imbalance (minor classes downsampled by factor rho)
to train_mask only; val/test unchanged. Sweeps rho in [1, 10, 50, 100].
Reports balanced accuracy along with acc/f1 via answer-ranking inference.
"""

from __future__ import annotations

import argparse
import collections
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance
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
    _pack_text_batch,
    _rank_answers,
)


RHO_LIST = [5, 10, 20]


@torch.no_grad()
def eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y, idx, label_names,
                llm_dtype, eval_batch_size=16, num_classes=None, return_per_class=False):
    if idx.numel() == 0:
        if return_per_class:
            return float("nan"), float("nan"), float("nan"), None
        return float("nan"), float("nan"), float("nan")
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
        preds.append(best_cls); trues.append(y[batch].cpu())
    preds = torch.cat(preds, dim=0); trues = torch.cat(trues, dim=0)
    acc = (preds == trues).float().mean().item() * 100.0
    f1 = f1_score(trues.numpy(), preds.numpy(), average="macro") * 100.0
    bacc = balanced_accuracy_score(trues.numpy(), preds.numpy()) * 100.0
    if return_per_class:
        nc = num_classes if num_classes is not None else len(label_names)
        per_class = recall_score(
            trues.numpy(), preds.numpy(),
            labels=list(range(nc)), average=None, zero_division=0,
        ) * 100.0
        return acc, f1, bacc, per_class
    return acc, f1, bacc


def _train_one(model, tokenizer, prompt_text, feat, nd_indices, y,
               train_idx, val_idx, label_names, llm_dtype, args):
    prompt_ids = _build_prompt_only_ids(tokenizer, prompt_text)
    optim = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=args.wd)
    device = next(model.parameters()).device

    val_pool = val_idx
    if args.val_subsample > 0 and val_pool.numel() > args.val_subsample:
        g = torch.Generator().manual_seed(42)
        val_pool = val_pool[torch.randperm(val_pool.numel(), generator=g)[:args.val_subsample]]

    best_val, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = torch.randperm(train_idx.numel())
        tp = train_idx[perm]
        total_loss, n_b = 0.0, 0
        for start in range(0, tp.numel(), args.batch_size):
            end = min(start + args.batch_size, tp.numel())
            batch = tp[start:end]
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
            val_acc, _, _ = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                        val_pool, label_names, llm_dtype, args.eval_batch_size)
            avg = total_loss / max(n_b, 1)
            print(f"[IMB-GEN] epoch={epoch} train_loss={avg:.4f} val_acc={val_acc:.4f}", flush=True)
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
    p.add_argument("--train_subsample", default=0, type=int,
                   help="Cap imbalanced train set to this size after step_imbalance (0 = no cap). "
                        "Stratifies the subsample so minor classes keep their proportional share.")
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-IMB] device={device} llm={args.llm_name}")

    data, _ = load_tag(args.dataset)
    feat = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long() if data.y.dim() > 1 else data.y.long()
    y = y.to(device)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data)
    if not label_names:
        raise RuntimeError(f"no label_names for {args.dataset}")
    splits = _get_splits(data, N)
    print(f"[LLAGA-IMB] {args.dataset}: N={N} C={len(label_names)}", flush=True)

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
    for split_idx, split in enumerate(splits[: args.num_splits]):
        torch.manual_seed(split_idx); np.random.seed(split_idx)

        print(f"[LLAGA-IMB] split={split_idx}: building ND...", flush=True)
        nd_indices = build_nd_subgraph_indices(
            edge_index, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size, seed=split_idx,
        )

        orig_train = split["train"].bool()
        val_mask = split["val"].bool()
        test_mask = split["test"].bool()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1).cpu()
        test_idx_full = torch.nonzero(test_mask, as_tuple=False).squeeze(-1).cpu()
        if args.test_subsample > 0 and test_idx_full.numel() > args.test_subsample:
            g = torch.Generator().manual_seed(split_idx * 9973 + 1)
            test_idx = test_idx_full[torch.randperm(test_idx_full.numel(), generator=g)[:args.test_subsample]]
        else:
            test_idx = test_idx_full

        llm_dtype = None

        num_classes = len(label_names)
        for rho in RHO_LIST:
            # Apply imbalance
            if rho == 1:
                tr_mask_rho = orig_train
                meta = {"minor_classes": []}
            else:
                tr_mask_rho, meta = make_step_imbalance(
                    orig_train.cpu(), y.cpu(), rho=rho, seed=split_idx,
                )
            minor_classes = sorted(meta.get("minor_classes", []))
            tr_idx_rho = torch.nonzero(tr_mask_rho, as_tuple=False).squeeze(-1).cpu()
            if tr_idx_rho.numel() == 0:
                print(f"[LLAGA-IMB] split={split_idx} rho={rho}: empty train, skip", flush=True)
                continue
            # Optional train cap: stratified per-class proportional subsample so minor
            # classes keep their proportional share after rho downsampling.
            if args.train_subsample > 0 and tr_idx_rho.numel() > args.train_subsample:
                rng = np.random.RandomState(split_idx * 7919 + rho)
                tr_y = y[tr_idx_rho].cpu().numpy()
                cap = args.train_subsample
                total = tr_idx_rho.numel()
                kept = []
                # Per-class proportional cap; ensure each class keeps at least 1
                for c in range(num_classes):
                    cls_idx = tr_idx_rho[tr_y == c].cpu().numpy()
                    if len(cls_idx) == 0:
                        continue
                    n_keep = max(1, int(round(len(cls_idx) * cap / total)))
                    if n_keep >= len(cls_idx):
                        kept.append(cls_idx)
                    else:
                        sel = rng.choice(len(cls_idx), size=n_keep, replace=False)
                        kept.append(cls_idx[sel])
                tr_idx_rho = torch.as_tensor(np.concatenate(kept), dtype=torch.long)
            print(f"[LLAGA-IMB] split={split_idx} rho={rho}: train={tr_idx_rho.numel()} "
                  f"minor_classes={minor_classes}", flush=True)

            # Fresh model
            model = LlagaGenerative(cfg).to(device)
            if args.projector_ckpt:
                state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
                model.projector.load_state_dict(state, strict=True)
            if llm_dtype is None:
                llm_dtype = next(model.llm.parameters()).dtype

            _train_one(model, tokenizer, prompt_text, feat, nd_indices, y,
                       tr_idx_rho, val_idx, label_names, llm_dtype, args)

            acc, f1, bacc, per_class = eval_ranked(
                model, tokenizer, prompt_ids, feat, nd_indices, y,
                test_idx, label_names, llm_dtype, args.eval_batch_size,
                num_classes=num_classes, return_per_class=True,
            )
            print(f"[IMB_RAW] method=LLaGA_GEN dataset={args.dataset} split_idx={split_idx} "
                  f"seed={split_idx} rho={rho} test_acc={acc:.4f} macro_f1={f1:.4f} "
                  f"balanced_acc={bacc:.4f}", flush=True)
            if per_class is not None:
                minor_str = ",".join(str(c) for c in minor_classes)
                recall_str = ",".join(f"{r:.4f}" for r in per_class.tolist())
                print(f"[IMB_PER_CLASS] method=LLaGA dataset={args.dataset} "
                      f"rho={rho} rep={split_idx} "
                      f"minor_classes=[{minor_str}] per_class_recall=[{recall_str}]",
                      flush=True)
            all_results.append({"split": split_idx, "rho": rho,
                                "acc": acc, "f1": f1, "bacc": bacc})

            del model; torch.cuda.empty_cache()

    print(f"\n=== LLaGA GEN Imbalance ({args.dataset}) ===")
    ga = collections.defaultdict(list); gf = collections.defaultdict(list); gb = collections.defaultdict(list)
    for r in all_results:
        ga[r["rho"]].append(r["acc"]); gf[r["rho"]].append(r["f1"]); gb[r["rho"]].append(r["bacc"])
    parts_a, parts_f, parts_b = [], [], []
    for rho in sorted(ga.keys()):
        a = np.array(ga[rho]); f = np.array(gf[rho]); b = np.array(gb[rho])
        parts_a.append(f'rho{rho}="{a.mean():.2f} ± {a.std():.2f}"')
        parts_f.append(f'rho{rho}="{f.mean():.2f} ± {f.std():.2f}"')
        parts_b.append(f'rho{rho}="{b.mean():.2f} ± {b.std():.2f}"')
    print(f"[IMB_AGG_OLD] method=LLaGA_GEN dataset={args.dataset} metric=acc " + " ".join(parts_a))
    print(f"[IMB_AGG_OLD] method=LLaGA_GEN dataset={args.dataset} metric=f1 " + " ".join(parts_f))
    print(f"[IMB_AGG_OLD] method=LLaGA_GEN dataset={args.dataset} metric=bacc " + " ".join(parts_b))

    # Per-rho IMB_AGG line in the format `_aggregate_from_logs.py` parses.
    for rho in sorted(ga.keys()):
        a = np.array(ga[rho]); f = np.array(gf[rho]); b = np.array(gb[rho])
        n_reps = len(a)
        # Use sample std (ddof=1) when n >= 2, else 0.0 — matches sklearn/numpy default downstream.
        a_s = float(a.std(ddof=1)) if n_reps >= 2 else 0.0
        f_s = float(f.std(ddof=1)) if n_reps >= 2 else 0.0
        b_s = float(b.std(ddof=1)) if n_reps >= 2 else 0.0
        print(
            f'[IMB_AGG] method=LLaGA dataset={args.dataset} rho={rho} '
            f'n_reps={n_reps} '
            f'bacc="{b.mean():.2f} ± {b_s:.2f}" '
            f'macro_f1="{f.mean():.2f} ± {f_s:.2f}" '
            f'acc="{a.mean():.2f} ± {a_s:.2f}"',
            flush=True,
        )


if __name__ == "__main__":
    main()
