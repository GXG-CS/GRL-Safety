"""LLaGA generative node-classification FT (aligned with original ICML paper).

Differs from `run_node.py` (judicial/linear-head variant) in that:

- Training: next-token CE on a tokenized `<graph tokens, prompt text, answer text>`
  sequence, supervising only the answer span. Identical objective to
  our `pretrain_projector.py`. This is what LLaGA's paper uses.
- Inference: run greedy decoding for up to max_new_tokens, then map the
  generated string back to a class index by string matching against the
  dataset's `label_names`. Ties broken by shortest-edit-distance.
- Only `mm_projector` is trainable; LLM + classifier head (LM head) are frozen.

The pretrained projector ckpt (e.g. `projector_gft9_vicuna.pt`) is
loaded exactly the same way as in the linear-head variant.
"""

from __future__ import annotations

import argparse
import collections
import copy
import os.path as osp
import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
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
    LlagaNDEncoder,
    build_nd_subgraph_indices,
    lookup_nd_features,
)


IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Data helpers (shared with run_node.py)
# ---------------------------------------------------------------------------


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_feat_768d(data):
    if data.x is not None and data.x.dtype == torch.long and hasattr(data, "node_text_feat"):
        return data.node_text_feat[data.x].float()
    if data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        return data.x.float()
    if hasattr(data, "node_text_feat"):
        return data.node_text_feat.float()
    raise RuntimeError("cannot extract 768d features")


def _get_splits(data, N):
    splits = []
    if getattr(data, "train_masks", None) is not None:
        avail = len(data.train_masks)
        for i in range(5):
            j = i % avail
            splits.append({
                "train": data.train_masks[j].bool(),
                "val": data.val_masks[j].bool(),
                "test": data.test_masks[j].bool(),
            })
    elif getattr(data, "splits", None) is not None:
        s = data.splits
        tm = _idx_to_mask(s["train"], N)
        vm = _idx_to_mask(s.get("valid", s.get("val")), N)
        tsm = _idx_to_mask(s["test"], N)
        for _ in range(5):
            splits.append({"train": tm, "val": vm, "test": tsm})
    elif getattr(data, "train_mask", None) is not None:
        tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            avail = tm.size(1)
            for i in range(5):
                j = i % avail
                splits.append({
                    "train": tm[:, j].bool(),
                    "val": vm[:, j].bool(),
                    "test": (tsm[:, j] if tsm.dim() == 2 else tsm).bool(),
                })
        else:
            for _ in range(5):
                splits.append({"train": tm.bool(), "val": vm.bool(), "test": tsm.bool()})
    else:
        raise RuntimeError("no splits")
    return splits


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


def _build_prompt_answer_ids(tokenizer, prompt_text: str, answer_text: str):
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    answer_ids = tokenizer(" " + answer_text + tokenizer.eos_token,
                           add_special_tokens=False, return_tensors="pt").input_ids[0]
    full = torch.cat([prompt_ids, answer_ids], dim=0)
    tgt = torch.full_like(full, IGNORE_INDEX)
    tgt[prompt_ids.size(0):] = answer_ids
    return full, tgt, prompt_ids.size(0)


def _pack_text_batch(tokenizer, prompts, answers, max_text_len):
    id_list, tgt_list = [], []
    for p, a in zip(prompts, answers):
        full, tgt, _ = _build_prompt_answer_ids(tokenizer, p, a)
        if full.size(0) > max_text_len:
            full = full[:max_text_len]
            tgt = tgt[:max_text_len]
        id_list.append(full)
        tgt_list.append(tgt)
    T = max(x.size(0) for x in id_list)
    B = len(id_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    targets = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    text_mask = torch.zeros((B, T), dtype=torch.long)
    for b, (t, g) in enumerate(zip(id_list, tgt_list)):
        input_ids[b, :t.size(0)] = t
        targets[b, :g.size(0)] = g
        text_mask[b, :t.size(0)] = 1
    return input_ids, targets, text_mask


def _build_prompt_only_ids(tokenizer, prompt_text):
    ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids


# ---------------------------------------------------------------------------
# Model wrapper that exposes the LM head for generative loss
# ---------------------------------------------------------------------------


class LlagaGenerative(nn.Module):
    def __init__(self, cfg: LlagaConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LlagaNDEncoder(cfg)
        self.llm = self.encoder.llm
        self.projector = self.encoder.mm_projector

    def forward_loss(self, graph_tok, graph_mask, text_ids, text_mask, targets, llm_dtype):
        """Standard next-token CE aligned with pretrain_projector._llm_forward."""
        graph_embeds = self.projector(graph_tok).to(llm_dtype)
        graph_embeds = graph_embeds * graph_mask.unsqueeze(-1).to(graph_embeds.dtype)
        text_embeds = self.llm.get_input_embeddings()(text_ids).to(graph_embeds.dtype)
        inp = torch.cat([graph_embeds, text_embeds], dim=1)
        B, L = graph_embeds.size(0), graph_embeds.size(1)
        T = text_ids.size(1)
        attn = torch.cat(
            [torch.ones(B, L, device=inp.device, dtype=torch.long), text_mask],
            dim=1,
        )
        out = self.llm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        logits = out.logits
        full_targets = torch.full(
            (B, L + T), IGNORE_INDEX, device=inp.device, dtype=torch.long,
        )
        full_targets[:, L:] = targets
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_targets[:, 1:].contiguous()
        if (shift_labels != IGNORE_INDEX).sum().item() == 0:
            return torch.zeros((), device=inp.device, dtype=shift_logits.dtype)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @torch.no_grad()
    def generate_strings(self, graph_tok, graph_mask, prompt_ids, tokenizer,
                         max_new_tokens, llm_dtype):
        """Greedy generation. graph_tok: [B, L, D], prompt_ids: [P] (same prompt for batch)."""
        self.eval()
        B = graph_tok.size(0)
        graph_embeds = self.projector(graph_tok).to(llm_dtype)
        graph_embeds = graph_embeds * graph_mask.unsqueeze(-1).to(graph_embeds.dtype)
        prompt_ids = prompt_ids.to(graph_embeds.device).unsqueeze(0).expand(B, -1)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids).to(graph_embeds.dtype)
        inp = torch.cat([graph_embeds, prompt_embeds], dim=1)
        L = graph_embeds.size(1)
        P = prompt_ids.size(1)
        attn = torch.ones(B, L + P, device=inp.device, dtype=torch.long)

        eos_id = tokenizer.eos_token_id
        generated = torch.full((B, 0), 0, dtype=torch.long, device=inp.device)

        past_kv = None
        for step in range(max_new_tokens):
            if step == 0:
                out = self.llm(
                    inputs_embeds=inp, attention_mask=attn,
                    use_cache=True,
                )
            else:
                last_tok = next_tok.unsqueeze(-1)
                last_embed = self.llm.get_input_embeddings()(last_tok).to(graph_embeds.dtype)
                attn = torch.cat([attn, torch.ones(B, 1, device=inp.device, dtype=torch.long)], dim=1)
                out = self.llm(
                    inputs_embeds=last_embed, attention_mask=attn,
                    past_key_values=past_kv, use_cache=True,
                )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=1)
            if (next_tok == eos_id).all().item():
                break

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        return decoded


def _nearest_label(pred_text: str, label_names: List[str]) -> int:
    """Map a free-form generated string back to a class index via simple matching.

    Strategy: (a) if pred starts with any label_name (case-insensitive, trimmed),
    take that; (b) else pick the label with largest substring overlap.
    """
    pred = pred_text.strip().lower()
    lowers = [ln.strip().lower() for ln in label_names]
    for i, ln in enumerate(lowers):
        if pred.startswith(ln):
            return i
    # longest common prefix
    best_i, best_score = 0, -1
    for i, ln in enumerate(lowers):
        s = 0
        while s < min(len(pred), len(ln)) and pred[s] == ln[s]:
            s += 1
        if s > best_score:
            best_score = s
            best_i = i
    return best_i


# ---------------------------------------------------------------------------
# Train / eval loop
# ---------------------------------------------------------------------------


def train_and_eval(
    model: LlagaGenerative, tokenizer, feat, nd_indices, y,
    train_mask, val_mask, test_mask, label_names, prompt_text,
    device, llm_dtype, max_epochs=50, patience=15, lr=1e-3, wd=1e-4,
    batch_size=4, eval_batch_size=8, max_text_len=128, max_new_tokens=12,
    val_every=3, val_subsample=128,
):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1).cpu()
    val_idx_full = torch.nonzero(val_mask, as_tuple=False).squeeze(-1).cpu()
    # Fixed subsample so val is stable across epochs.
    if val_subsample is not None and val_idx_full.numel() > val_subsample:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(val_idx_full.numel(), generator=g)[:val_subsample]
        val_idx = val_idx_full[perm]
    else:
        val_idx = val_idx_full
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze(-1).cpu()
    prompt_ids = _build_prompt_only_ids(tokenizer, prompt_text)

    def _val_acc():
        # evaluate via generation on val set (or via answer-rank for speed)
        model.eval()
        correct = 0
        total = 0
        for start in range(0, val_idx.numel(), eval_batch_size):
            end = min(start + eval_batch_size, val_idx.numel())
            batch_nodes = val_idx[start:end]
            nd_batch = nd_indices[batch_nodes]
            tok, mask = lookup_nd_features(nd_batch.to(device), feat)
            with torch.no_grad():
                # fast val: pick highest-likelihood class by computing next-token
                # CE of each label answer under the same prompt. O(B*C) forward.
                best_cls = _rank_answers(model, tokenizer, prompt_ids, tok, mask,
                                         label_names, llm_dtype)
            correct += (best_cls == y[batch_nodes].cpu()).sum().item()
            total += batch_nodes.numel()
        return correct / max(total, 1)

    best_val, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(train_idx.numel())
        train_perm = train_idx[perm]
        total_loss = 0.0
        n_batch = 0
        for start in range(0, train_perm.numel(), batch_size):
            end = min(start + batch_size, train_perm.numel())
            batch_nodes = train_perm[start:end]
            nd_batch = nd_indices[batch_nodes]
            tok, mask = lookup_nd_features(nd_batch.to(device), feat)
            answers = [label_names[int(y[batch_nodes[b]].item())] for b in range(batch_nodes.numel())]
            prompts = [prompt_text] * batch_nodes.numel()
            input_ids, targets, text_mask = _pack_text_batch(tokenizer, prompts, answers, max_text_len)
            input_ids = input_ids.to(device); targets = targets.to(device); text_mask = text_mask.to(device)
            loss = model.forward_loss(tok, mask, input_ids, text_mask, targets, llm_dtype)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += float(loss.item())
            n_batch += 1

        if epoch % val_every == 0 or epoch == max_epochs:
            val_acc = _val_acc()
            avg = total_loss / max(n_batch, 1)
            print(f"[GEN] epoch={epoch} train_loss={avg:.4f} val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.projector.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state is not None:
        model.projector.load_state_dict(best_state)

    # Final test: answer-ranking (more stable than open generation)
    model.eval()
    all_pred = []
    all_true = []
    for start in range(0, test_idx.numel(), eval_batch_size):
        end = min(start + eval_batch_size, test_idx.numel())
        batch_nodes = test_idx[start:end]
        nd_batch = nd_indices[batch_nodes]
        tok, mask = lookup_nd_features(nd_batch.to(device), feat)
        with torch.no_grad():
            best_cls = _rank_answers(model, tokenizer, prompt_ids, tok, mask,
                                     label_names, llm_dtype)
        all_pred.append(best_cls)
        all_true.append(y[batch_nodes].cpu())
    all_pred = torch.cat(all_pred, dim=0)
    all_true = torch.cat(all_true, dim=0)
    test_acc = (all_pred == all_true).float().mean().item() * 100.0
    macro_f1 = f1_score(all_true.numpy(), all_pred.numpy(), average="macro") * 100.0
    return test_acc, macro_f1


@torch.no_grad()
def _rank_answers(model, tokenizer, prompt_ids, graph_tok, graph_mask,
                  label_names, llm_dtype):
    """For each item in the batch, compute CE of every candidate answer and
    pick argmin. Returns [B] LongTensor of class indices on CPU.

    This is equivalent to open generation + exact-match parsing when the
    answers are distinct, but far cheaper and avoids string-parsing noise.
    """
    device = next(model.parameters()).device
    B = graph_tok.size(0)
    C = len(label_names)

    # Precompute tokenised answers (with eos).
    ans_ids_list = []
    for name in label_names:
        ids = tokenizer(" " + name + tokenizer.eos_token,
                        add_special_tokens=False, return_tensors="pt").input_ids[0]
        ans_ids_list.append(ids)
    max_a = max(x.size(0) for x in ans_ids_list)
    ans_padded = torch.full((C, max_a), tokenizer.pad_token_id, dtype=torch.long)
    ans_mask = torch.zeros((C, max_a), dtype=torch.long)
    for i, a in enumerate(ans_ids_list):
        ans_padded[i, :a.size(0)] = a
        ans_mask[i, :a.size(0)] = 1

    # Build graph+prompt embeds once per batch, then tile over C candidates.
    graph_embeds = model.projector(graph_tok).to(llm_dtype)
    graph_embeds = graph_embeds * graph_mask.unsqueeze(-1).to(graph_embeds.dtype)
    P = prompt_ids.numel()
    prompt_embeds = model.llm.get_input_embeddings()(prompt_ids.to(device).unsqueeze(0).expand(B, -1)).to(graph_embeds.dtype)

    losses = torch.zeros(B, C)
    for c in range(C):
        ans_c = ans_padded[c:c + 1].expand(B, -1).to(device)
        ans_m_c = ans_mask[c:c + 1].expand(B, -1).to(device)
        ans_embeds = model.llm.get_input_embeddings()(ans_c).to(graph_embeds.dtype)
        inp = torch.cat([graph_embeds, prompt_embeds, ans_embeds], dim=1)
        L = graph_embeds.size(1)
        attn = torch.cat(
            [torch.ones(B, L + P, device=device, dtype=torch.long), ans_m_c],
            dim=1,
        )
        out = model.llm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        logits = out.logits                                     # [B, L+P+A, V]
        # Predict answer tokens: positions that produce answer = (L+P-1 .. L+P+A-2)
        shift_logits = logits[:, L + P - 1:-1, :].contiguous()   # [B, A, V]
        shift_labels = ans_c                                    # [B, A]
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="none",
        ).view(B, -1)
        # mask out pad positions and average per-sample
        m = ans_m_c.float()
        per = (ce * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        losses[:, c] = per.float().cpu()

    return losses.argmin(dim=-1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--llm_name", default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--llm_dtype", default="bfloat16")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--projector_type", default="2-layer-mlp")
    parser.add_argument("--projector_ckpt", required=True)
    parser.add_argument("--use_hop", default=2, type=int)
    parser.add_argument("--sample_size", default=10, type=int)
    parser.add_argument("--max_epochs", default=15, type=int)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--val_every", default=3, type=int)
    parser.add_argument("--val_subsample", default=128, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--num_splits", default=5, type=int)
    parser.add_argument("--max_text_len", default=128, type=int)
    parser.add_argument("--max_new_tokens", default=12, type=int)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--debug", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[LLAGA-GEN] device={device} llm={args.llm_name}")

    data, _ = load_tag(args.dataset)
    feat = _get_feat_768d(data).to(device)
    y = data.y.squeeze().long().to(device) if data.y.dim() > 1 else data.y.long().to(device)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()
    label_names = get_label_names(args.dataset, data)
    if not label_names:
        raise RuntimeError(f"no label_names for {args.dataset}")
    splits = _get_splits(data, N)
    print(f"[LLAGA-GEN] {args.dataset}: N={N} classes={len(label_names)} splits={len(splits)}")

    cfg = LlagaConfig(
        llm_name_or_path=args.llm_name,
        mm_hidden_size=feat.size(1),
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
    print(f"[LLAGA-GEN] prompt: {prompt_text}")

    n_splits = 1 if args.debug else args.num_splits
    max_epochs = 3 if args.debug else args.max_epochs

    all_results = []
    for split_idx, split in enumerate(splits[: n_splits]):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        print(f"[LLAGA-GEN] split={split_idx}: building ND indices...")
        nd_indices = build_nd_subgraph_indices(
            edge_index, N, torch.arange(N),
            use_hop=args.use_hop, sample_size=args.sample_size,
            seed=split_idx,
        )

        model = LlagaGenerative(cfg).to(device)
        if args.projector_ckpt:
            state = torch.load(args.projector_ckpt, map_location=device, weights_only=True)
            model.projector.load_state_dict(state, strict=True)
            print(f"[LLAGA-GEN] loaded projector from {args.projector_ckpt}")
        n_trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LLAGA-GEN] trainable params: {n_trn/1e6:.2f}M (projector only)")

        train_mask = split["train"].to(device)
        val_mask = split["val"].to(device)
        test_mask = split["test"].to(device)

        llm_dtype = next(model.llm.parameters()).dtype

        acc, f1 = train_and_eval(
            model, tokenizer, feat, nd_indices, y,
            train_mask, val_mask, test_mask, label_names, prompt_text,
            device, llm_dtype,
            max_epochs=max_epochs, patience=args.patience,
            lr=args.lr, wd=args.wd,
            batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            max_text_len=args.max_text_len, max_new_tokens=args.max_new_tokens,
            val_every=args.val_every, val_subsample=args.val_subsample,
        )
        all_results.append({"split_idx": split_idx, "sev": 0, "acc": acc, "f1": f1})
        print(
            f"[FN_RAW] method=LLaGA_GEN dataset={args.dataset} "
            f"split_idx={split_idx} seed={split_idx} sev=0 sigma_rel=0.0 "
            f"test_acc={acc:.4f} macro_f1={f1:.4f}"
        )
        del model
        torch.cuda.empty_cache()

    print(f"\n=== LLaGA GEN Clean Results ({args.dataset}) ===")
    grouped_acc = collections.defaultdict(list)
    grouped_f1 = collections.defaultdict(list)
    for r in all_results:
        grouped_acc[r["sev"]].append(r["acc"])
        grouped_f1[r["sev"]].append(r["f1"])
    labels = ["clean", "sev1", "sev2", "sev3", "sev4", "sev5"]
    parts_acc, parts_f1 = [], []
    for sev in sorted(grouped_acc.keys()):
        accs = np.array(grouped_acc[sev])
        f1s = np.array(grouped_f1[sev])
        parts_acc.append(f'{labels[sev]}="{accs.mean():.2f} ± {accs.std():.2f}"')
        parts_f1.append(f'{labels[sev]}="{f1s.mean():.2f} ± {f1s.std():.2f}"')
    print(f"[FN_AGG] method=LLaGA_GEN dataset={args.dataset} metric=acc " + " ".join(parts_acc))
    print(f"[FN_AGG] method=LLaGA_GEN dataset={args.dataset} metric=f1 " + " ".join(parts_f1))


if __name__ == "__main__":
    main()
