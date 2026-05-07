"""LLaGA fairness eval on tolokers — v2 with minority-oversampling fix.

Differences from run_fairness.py:
  - Oversample minority class (y=1) so each training epoch sees a roughly
    balanced positive/negative ratio. Prevents mode-collapse on tolokers
    where banned-rate ~22%.
  - Validation early-stops on macro-F1 (not accuracy) so the optimum is
    not the trivial majority predictor.
  - Boosted default lr (3e-3) and patience (8) to give the projector
    more chance to learn the minority signal.
"""

from __future__ import annotations

import argparse
import os.path as osp
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import load_tolokers_education_binary, compute_group_fairness
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


@torch.no_grad()
def predict_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y, idx, label_names,
                   llm_dtype, eval_batch_size=16):
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
    return preds, trues


def build_oversampled_train(train_idx_cpu, y_cpu, ratio_target=1.0):
    """Return a permuted train-index tensor where minority class is
    upsampled to roughly match majority count."""
    y_train = y_cpu[train_idx_cpu].long()
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return train_idx_cpu[torch.randperm(train_idx_cpu.numel())]
    pos_idx = train_idx_cpu[y_train == 1]
    neg_idx = train_idx_cpu[y_train == 0]
    # Decide which is minority and how many copies needed
    if n_pos < n_neg:
        target_n_pos = int(n_neg * ratio_target)
        repeat = max(1, (target_n_pos + n_pos - 1) // n_pos)
        pos_idx = pos_idx.repeat(repeat)[:target_n_pos]
    else:
        target_n_neg = int(n_pos * ratio_target)
        repeat = max(1, (target_n_neg + n_neg - 1) // n_neg)
        neg_idx = neg_idx.repeat(repeat)[:target_n_neg]
    full = torch.cat([pos_idx, neg_idx])
    return full[torch.randperm(full.numel())]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="tolokers")
    p.add_argument("--llm_name", default="lmsys/vicuna-7b-v1.5-16k")
    p.add_argument("--llm_dtype", default="bfloat16")
    p.add_argument("--attn_impl", default="eager")
    p.add_argument("--projector_type", default="2-layer-mlp")
    p.add_argument("--projector_ckpt", required=True)
    p.add_argument("--use_hop", default=2, type=int)
    p.add_argument("--sample_size", default=10, type=int)
    p.add_argument("--max_epochs", default=80, type=int)
    p.add_argument("--patience", default=8, type=int)
    p.add_argument("--val_every", default=2, type=int)
    p.add_argument("--val_subsample", default=256, type=int)
    p.add_argument("--lr", default=3e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--eval_batch_size", default=16, type=int)
    p.add_argument("--num_splits", default=5, type=int)
    p.add_argument("--test_subsample", default=-1, type=int)
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--oversample_ratio", default=1.0, type=float,
                   help="target minority/majority ratio after oversampling")
    args = p.parse_args()

    if args.dataset != "tolokers":
        raise NotImplementedError("fairness only on tolokers")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-FAIR-V2] device={device} llm={args.llm_name}", flush=True)

    data, _ = load_tag(args.dataset)
    feat = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long() if data.y.dim() > 1 else data.y.long()
    y_cpu = y.cpu()
    y = y.to(device)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data) or ["0", "1"]
    splits = _get_splits(data, N)

    sensitive, sens_meta = load_tolokers_education_binary()
    print(f"[LLAGA-FAIR-V2] tolokers N={N} class-1-rate={(y_cpu==1).float().mean().item():.4f} "
          f"sensitive: {sens_meta}", flush=True)

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

    all_metrics = []
    for split_idx, split in enumerate(splits[: args.num_splits]):
        torch.manual_seed(split_idx); np.random.seed(split_idx)
        print(f"[LLAGA-FAIR-V2] split={split_idx}: building ND...", flush=True)
        nd_indices = build_nd_subgraph_indices(
            edge_index, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size, seed=split_idx,
        )

        train_mask = split["train"].bool()
        val_mask = split["val"].bool()
        test_mask = split["test"].bool()
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1).cpu()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1).cpu()
        test_idx_full = torch.nonzero(test_mask, as_tuple=False).squeeze(-1).cpu()
        test_idx = test_idx_full
        if args.test_subsample > 0 and test_idx_full.numel() > args.test_subsample:
            g = torch.Generator().manual_seed(split_idx * 9973 + 1)
            test_idx = test_idx_full[torch.randperm(test_idx_full.numel(), generator=g)[:args.test_subsample]]

        # Class balance in train
        yt = y_cpu[train_idx]
        print(f"[LLAGA-FAIR-V2] split={split_idx} train pos={int((yt==1).sum())} "
              f"neg={int((yt==0).sum())}", flush=True)

        model = LlagaGenerative(cfg).to(device)
        if args.projector_ckpt:
            state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
            model.projector.load_state_dict(state, strict=True)
        optim = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=args.wd)
        llm_dtype = next(model.llm.parameters()).dtype

        val_pool = val_idx
        if args.val_subsample > 0 and val_pool.numel() > args.val_subsample:
            g = torch.Generator().manual_seed(split_idx * 17 + 5)
            val_pool = val_pool[torch.randperm(val_pool.numel(), generator=g)[:args.val_subsample]]

        best_val_f1, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            tp = build_oversampled_train(train_idx, y_cpu, ratio_target=args.oversample_ratio)
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
                vp, vt = predict_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                        val_pool, label_names, llm_dtype, args.eval_batch_size)
                val_acc = (vp == vt).float().mean().item()
                val_f1 = f1_score(vt.numpy(), vp.numpy(), average="macro")
                pos_rate = (vp == 1).float().mean().item()
                avg = total_loss / max(n_b, 1)
                print(f"[FAIR-V2] split={split_idx} epoch={epoch} loss={avg:.4f} "
                      f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} pos_rate={pos_rate:.4f}",
                      flush=True)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        break

        if best_state is not None:
            model.projector.load_state_dict(best_state)

        preds, trues = predict_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                      test_idx, label_names, llm_dtype, args.eval_batch_size)
        preds_np = preds.numpy().astype(int)
        trues_np = trues.numpy().astype(int)
        score = preds_np.astype(float)
        sens_np = sensitive[test_idx].numpy().astype(int)
        test_mask_arr = np.ones_like(preds_np, dtype=bool)
        fm = compute_group_fairness(trues_np, preds_np, score, sens_np, test_mask=test_mask_arr)
        acc = (preds_np == trues_np).mean() * 100.0
        f1 = f1_score(trues_np, preds_np, average="macro") * 100.0
        pos_rate = (preds_np == 1).mean()
        print(f"[FAIR_RAW] method=LLaGA_GEN_V2 dataset={args.dataset} split_idx={split_idx} "
              f"seed={split_idx} test_acc={acc:.4f} macro_f1={f1:.4f} pos_rate={pos_rate:.4f} "
              f"delta_sp={fm.get('delta_sp', float('nan')):.4f} "
              f"delta_eo={fm.get('delta_eo', float('nan')):.4f} "
              f"delta_util={fm.get('delta_utility', float('nan')):.4f}", flush=True)
        all_metrics.append({"split": split_idx, "acc": acc, "f1": f1,
                            "delta_sp": fm.get("delta_sp", float("nan")),
                            "delta_eo": fm.get("delta_eo", float("nan")),
                            "delta_util": fm.get("delta_utility", float("nan"))})

        del model; torch.cuda.empty_cache()

    print(f"\n=== LLaGA GEN-V2 Fairness ({args.dataset}) ===")
    def _agg(k):
        arr = np.array([r[k] for r in all_metrics])
        return f"{np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}"
    print(f"[FAIR_AGG] method=LLaGA_GEN_V2 dataset={args.dataset} "
          f"acc=\"{_agg('acc')}\" macro_f1=\"{_agg('f1')}\" "
          f"delta_sp=\"{_agg('delta_sp')}\" delta_eo=\"{_agg('delta_eo')}\" "
          f"delta_util=\"{_agg('delta_util')}\"", flush=True)


if __name__ == "__main__":
    main()
