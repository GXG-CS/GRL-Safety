"""LLaGA generative FT + structural (degree-based) fairness eval.

Trains LLaGA node-classification on a TAG dataset, then computes head/tail
accuracy gap based on test-node degree. Reports per-split mean+std.
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
from sgb.metrics.fairness import compute_structural_fairness
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
    p.add_argument("--test_subsample", default=-1, type=int)
    p.add_argument("--train_subsample", default=-1, type=int,
                   help="cap train pool size per split (default -1 = no cap)")
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--q", default=0.2, type=float,
                   help="quantile for head/tail split (0.2 = top/bottom 20%)")
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-STRUCT] device={device} llm={args.llm_name}", flush=True)

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

    # Per-node degree (for head/tail split)
    degree = torch.bincount(edge_index[0], minlength=N).cpu().numpy()
    print(f"[LLAGA-STRUCT] {args.dataset}: N={N} C={len(label_names)} deg mean={degree.mean():.1f}",
          flush=True)

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
        print(f"[LLAGA-STRUCT] split={split_idx}: building ND...", flush=True)
        nd_indices = build_nd_subgraph_indices(
            edge_index, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size, seed=split_idx,
        )

        train_mask = split["train"].bool()
        val_mask = split["val"].bool()
        test_mask = split["test"].bool()
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1).cpu()
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1).cpu()
        if args.train_subsample > 0 and train_idx.numel() > args.train_subsample:
            g = torch.Generator().manual_seed(split_idx * 7919 + 3)
            train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)[:args.train_subsample]]
            print(f"[LLAGA-STRUCT] split={split_idx} train subsampled to {train_idx.numel()}", flush=True)
        test_idx_full = torch.nonzero(test_mask, as_tuple=False).squeeze(-1).cpu()
        if args.test_subsample > 0 and test_idx_full.numel() > args.test_subsample:
            g = torch.Generator().manual_seed(split_idx * 9973 + 1)
            test_idx = test_idx_full[torch.randperm(test_idx_full.numel(), generator=g)[:args.test_subsample]]
        else:
            test_idx = test_idx_full

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
                vp, vt = predict_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                        val_pool, label_names, llm_dtype, args.eval_batch_size)
                val_acc = (vp == vt).float().mean().item()
                avg = total_loss / max(n_b, 1)
                print(f"[STRUCT] split={split_idx} epoch={epoch} loss={avg:.4f} val_acc={val_acc:.4f}",
                      flush=True)
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

        preds, trues = predict_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                      test_idx, label_names, llm_dtype, args.eval_batch_size)
        preds_np = preds.numpy().astype(int)
        trues_np = trues.numpy().astype(int)
        acc = (preds_np == trues_np).mean() * 100.0

        # Build full-N pred / true / mask for compute_structural_fairness
        full_pred = np.zeros(N, dtype=int)
        full_true = np.zeros(N, dtype=int)
        tsm_arr = np.zeros(N, dtype=bool)
        full_pred[test_idx.numpy()] = preds_np
        full_true[test_idx.numpy()] = trues_np
        tsm_arr[test_idx.numpy()] = True
        struct = compute_structural_fairness(full_true, full_pred, degree, test_mask=tsm_arr, q=args.q)

        print(f"[STRUCT_RAW] method=LLaGA dataset={args.dataset} split={split_idx} "
              f"acc={acc:.4f} acc_head={struct['acc_head']:.4f} acc_tail={struct['acc_tail']:.4f} "
              f"acc_gap={struct['acc_gap']:.4f} f1_head={struct['f1_head']:.4f} f1_tail={struct['f1_tail']:.4f} "
              f"f1_gap={struct['f1_gap']:.4f} n_head={struct['n_head']} n_tail={struct['n_tail']} q={args.q}",
              flush=True)
        all_results.append({"split": split_idx, "acc": acc, **struct})
        del model; torch.cuda.empty_cache()

    print(f"\n=== LLaGA STRUCT-FAIR ({args.dataset}) ===")
    def _agg(k):
        vs = [r[k] for r in all_results
              if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs: return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))
    a, sa = _agg('acc'); ah, sah = _agg('acc_head'); at, sat = _agg('acc_tail'); ag, sag = _agg('acc_gap')
    fh, sfh = _agg('f1_head'); ft, sft = _agg('f1_tail'); fg, sfg = _agg('f1_gap')
    print(f"[STRUCT_AGG] method=LLaGA dataset={args.dataset} n_runs={len(all_results)} "
          f"acc=\"{a:.2f} ± {sa:.2f}\" acc_head=\"{ah:.2f} ± {sah:.2f}\" "
          f"acc_tail=\"{at:.2f} ± {sat:.2f}\" acc_gap=\"{ag:.2f} ± {sag:.2f}\" "
          f"f1_head=\"{fh:.2f} ± {sfh:.2f}\" f1_tail=\"{ft:.2f} ± {sft:.2f}\" "
          f"f1_gap=\"{fg:.2f} ± {sfg:.2f}\" q={args.q}", flush=True)


if __name__ == "__main__":
    main()
