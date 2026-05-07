"""LLaGA generative FT + temporal-shift OOD evaluation (arxiv only).

Aggressive split (matches OFA run_ood_time.py):
  - train: papers with year <= 2010 (90% train, 10% id_val)
  - test : papers with year >= 2017 (>=7-year gap)

Mirrors run_ood_degree.py training/inference path; only the index
builder is replaced with a year-based split. Output rows appended to
experiments/ood/time/results/llaga_ood_time.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
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


def aggressive_split(year: torch.Tensor, split_seed: int):
    y = year.squeeze().long()
    train_idx = (y <= 2010).nonzero(as_tuple=True)[0]
    test_idx = (y >= 2017).nonzero(as_tuple=True)[0]
    rng = np.random.RandomState(split_seed)
    perm = torch.as_tensor(rng.permutation(int(train_idx.numel())), dtype=torch.long)
    cut = int(0.9 * train_idx.numel())
    return {
        "train":   train_idx[perm[:cut]],
        "id_val":  train_idx[perm[cut:]],
        "ood_test": test_idx,
    }


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
    p.add_argument("--dataset", default="arxiv")
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
    p.add_argument("--num_seeds", default=5, type=int)
    p.add_argument("--test_subsample", default=1000, type=int)
    p.add_argument("--max_text_len", default=128, type=int)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--output_csv", default=None)
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-OOD-TIME] device={device} llm={args.llm_name}", flush=True)

    data, _ = load_tag(args.dataset)
    assert hasattr(data, "node_year"), "arxiv required: data.node_year missing"
    year = data.node_year
    feat = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long() if data.y.dim() > 1 else data.y.long()
    y = y.to(device)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data)
    if not label_names:
        raise RuntimeError(f"no label_names for {args.dataset}")
    num_classes = len(label_names)
    print(f"[LLAGA-OOD-TIME] {args.dataset}: N={N} C={num_classes}", flush=True)

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

    print(f"[LLAGA-OOD-TIME] building ND for all {N} nodes...", flush=True)
    nd_indices = build_nd_subgraph_indices(
        edge_index, N, torch.arange(N),
        use_hop=args.use_hop, sample_size=args.sample_size, seed=0,
    )

    def _cap(idx, cap, seed):
        if cap <= 0 or idx.numel() <= cap:
            return idx
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(idx.numel(), generator=g)[:cap]
        return idx[perm]

    all_results = []
    for si in range(args.num_seeds):
        torch.manual_seed(si); np.random.seed(si)

        sp = aggressive_split(year, si)
        train_idx_cpu = sp["train"]
        id_val_cpu = sp["id_val"]
        ood_test_cpu = sp["ood_test"]
        print(f"[LLAGA-OOD-TIME] seed={si} train={train_idx_cpu.numel()} "
              f"id_val={id_val_cpu.numel()} ood_test={ood_test_cpu.numel()}", flush=True)

        id_test_eval = _cap(id_val_cpu, args.test_subsample, si * 31 + 1)
        ood_test_eval = _cap(ood_test_cpu, args.test_subsample, si * 31 + 2)

        model = LlagaGenerative(cfg).to(device)
        if args.projector_ckpt:
            state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
            model.projector.load_state_dict(state, strict=True)
        optim = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=args.wd)
        llm_dtype = next(model.llm.parameters()).dtype

        if args.val_subsample > 0 and id_val_cpu.numel() > args.val_subsample:
            gv = torch.Generator().manual_seed(si * 17 + 5)
            val_eval = id_val_cpu[torch.randperm(id_val_cpu.numel(), generator=gv)[:args.val_subsample]]
        else:
            val_eval = id_val_cpu

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
                print(f"[OOD-TIME] seed={si} epoch={epoch} train_loss={avg:.4f} id_val={val_acc:.4f}", flush=True)
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

        id_acc, id_f1 = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                    id_test_eval, label_names, llm_dtype, args.eval_batch_size)
        ood_acc, ood_f1 = eval_ranked(model, tokenizer, prompt_ids, feat, nd_indices, y,
                                      ood_test_eval, label_names, llm_dtype, args.eval_batch_size)
        gap = id_acc - ood_acc if id_acc == id_acc and ood_acc == ood_acc else float("nan")
        print(f"[OOD_RAW] method=LLaGA dataset={args.dataset} protocol=aggressive seed={si} "
              f"id_test_acc={id_acc:.4f} id_test_f1={id_f1:.4f} "
              f"ood_test_acc={ood_acc:.4f} ood_test_f1={ood_f1:.4f} gap={gap:.4f}", flush=True)
        all_results.append({"dataset": args.dataset, "method": "llaga",
                            "protocol": "aggressive", "seed": si,
                            "val_acc": best_val,
                            "id_test_acc": id_acc, "ood_test_acc": ood_acc,
                            "gap": gap, "id_f1": id_f1, "ood_f1": ood_f1})

        del model; torch.cuda.empty_cache()

    print(f"\n=== LLaGA OOD-time ({args.dataset}, aggressive) ===")
    def _agg(k):
        arr = np.array([r[k] for r in all_results])
        return f"{np.nanmean(arr):.2f} ± {np.nanstd(arr):.2f}"
    print(f"[OOD_AGG] method=LLaGA dataset={args.dataset} protocol=aggressive "
          f"id=\"{_agg('id_test_acc')}\" ood=\"{_agg('ood_test_acc')}\" "
          f"gap=\"{_agg('gap')}\"", flush=True)

    out_csv = args.output_csv or osp.join(_PROJECT_ROOT, "experiments", "ood", "time",
                                          "results", "llaga_ood_time.csv")
    os.makedirs(osp.dirname(out_csv), exist_ok=True)
    new_file = not osp.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        if new_file: w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"[LLAGA-OOD-TIME] wrote {len(all_results)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
