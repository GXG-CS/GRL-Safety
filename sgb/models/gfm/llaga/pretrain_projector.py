"""Joint pretraining of the LLaGA mm_projector across GFT-9 via next-token CE.

Supports 3 task types so the projector is exposed to the same data
mixture GFT was pretrained on:

    task     datasets                       sample unit            answer
    ----     --------                       -----------            ------
    nc       cora, pubmed, wikics, arxiv    1 node ND subgraph     class name
    lp       WN18RR, FB15K237               2 node ND subgraphs    relation name
    gc       chemhiv, chempcba, chemblpre   full molecule atoms    yes / no

Only `mm_projector` is trained; LLM is frozen. We concatenate the
projected graph embeddings with the LLM's own text-token embeddings
and compute next-token CE only on answer positions.

Output:
    <ckpt_dir>/projector_<tag>.pt     state_dict of mm_projector
    <ckpt_dir>/projector_<tag>.json   config + training log summary
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.data.tag_registry import load as load_tag, DATASETS
from sgb.models.gfm.llaga.label_names import get_label_names
from sgb.models.gfm.llaga.model import (
    LlagaConfig,
    LlagaNDEncoder,
    build_nd_subgraph_indices,
    lookup_nd_features,
)


IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Text / tokenisation helpers
# ---------------------------------------------------------------------------


def _build_text_input_ids(tokenizer, prompt_text: str, answer_text: str):
    """Return (input_ids, target_ids). Prompt tokens map to IGNORE_INDEX."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    answer_ids = tokenizer(" " + answer_text + tokenizer.eos_token,
                           add_special_tokens=False, return_tensors="pt").input_ids[0]
    full = torch.cat([prompt_ids, answer_ids], dim=0)
    target = torch.full_like(full, IGNORE_INDEX)
    target[prompt_ids.size(0):] = answer_ids
    return full, target, prompt_ids.size(0)


def _pack_text_batch(tokenizer, prompts: List[str], answers: List[str], max_text_len: int):
    """Tokenise per-sample (prompt, answer); pad to batch max length.

    Returns:
        input_ids     [B, T]   long
        targets       [B, T]   long (IGNORE_INDEX on prompt positions)
        text_mask     [B, T]   long (1 = real token, 0 = pad)
    """
    id_list, tgt_list = [], []
    for p, a in zip(prompts, answers):
        full, tgt, _ = _build_text_input_ids(tokenizer, p, a)
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


def _get_feat_768d(data):
    """Node-level 768d features, regardless of whether data.x is an atom-index."""
    if data.x is not None and data.x.ndim == 2 and data.x.size(1) == 768:
        return data.x.float()
    if data.x is not None and data.x.dtype == torch.long and hasattr(data, "node_text_feat"):
        # chem-style: x is [N, 1] atom index into atom-vocab node_text_feat
        idx = data.x.squeeze(-1).long() if data.x.ndim == 2 else data.x.long()
        return data.node_text_feat[idx].float()
    if hasattr(data, "node_text_feat") and data.node_text_feat.size(0) == data.num_nodes:
        return data.node_text_feat.float()
    raise RuntimeError("cannot extract 768d per-node features")


# ---------------------------------------------------------------------------
# NC bundle
# ---------------------------------------------------------------------------


def _prepare_nc(name, device, use_hop, sample_size, seed, cap=None):
    data, _ = load_tag(name)
    feat = _get_feat_768d(data)
    y = data.y.squeeze().long() if data.y.dim() > 1 else data.y.long()
    N = feat.size(0)

    label_names = get_label_names(name, data)
    if not label_names:
        raise RuntimeError(f"{name}: no label_names available")

    if getattr(data, "train_masks", None) is not None:
        train_mask = data.train_masks[0].bool()
    elif getattr(data, "train_mask", None) is not None:
        train_mask = data.train_mask.bool()
        if train_mask.dim() > 1:
            train_mask = train_mask[:, 0].bool()
    else:
        g = torch.Generator().manual_seed(42)
        perm = torch.randperm(N, generator=g)
        train_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[perm[:int(0.6 * N)]] = True

    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
    if cap is not None and train_idx.numel() > cap:
        g = torch.Generator().manual_seed(seed * 100 + 1)
        perm = torch.randperm(train_idx.numel(), generator=g)[:cap]
        train_idx = train_idx[perm]
    nd = build_nd_subgraph_indices(
        data.edge_index.cpu(), N, train_idx,
        use_hop=use_hop, sample_size=sample_size, seed=seed,
    )
    return dict(
        name=name, task_type="nc",
        feat=feat.to(device),
        y=y.to(device),
        train_idx=train_idx,
        nd_indices=nd,
        label_names=label_names,
    )


def _build_batch_nc(bundle, sl, tokenizer, max_text_len):
    feat = bundle["feat"]
    nd = bundle["nd_indices"][sl]
    train_idx = bundle["train_idx"][sl]
    y = bundle["y"]
    label_names = bundle["label_names"]
    tok, graph_mask = lookup_nd_features(nd.to(feat.device), feat)

    # Omit enumeration of classes from the prompt: datasets like arxiv
    # (40 classes) and FB15K237 relation space (237 relations) easily blow
    # past max_text_len when listed, which truncates the answer tokens and
    # produces NaN CE.
    prompt = (
        f"Task: {bundle['name']} node classification. "
        f"Given the graph context, predict the class. Answer: "
    )
    prompts = [prompt] * len(sl)
    answers = [label_names[int(y[train_idx[b]].item())] for b in range(len(sl))]
    input_ids, targets, text_mask = _pack_text_batch(tokenizer, prompts, answers, max_text_len)

    return dict(
        tok=tok, graph_mask=graph_mask,
        input_ids=input_ids.to(feat.device),
        targets=targets.to(feat.device),
        text_mask=text_mask.to(feat.device),
    )


# ---------------------------------------------------------------------------
# LP bundle
# ---------------------------------------------------------------------------


def _prepare_lp(name, device, use_hop, sample_size, seed, cap=None):
    data, _ = load_tag(name)
    feat = _get_feat_768d(data)
    N = feat.size(0)
    edge_index = data.edge_index.cpu()           # [2, E_total]
    edge_types = data.edge_types.long()          # [E_total]

    # OFA exposes train_idx as indices into edge_index; confirm.
    train_idx = data.train_idx.long()
    if cap is not None and train_idx.numel() > cap:
        g = torch.Generator().manual_seed(seed * 100 + 2)
        perm = torch.randperm(train_idx.numel(), generator=g)[:cap]
        train_idx = train_idx[perm]
    rel_names = get_label_names(name, data)
    if not rel_names:
        raise RuntimeError(f"{name}: no relation names resolved")

    # For each training edge we need ND subgraphs of BOTH endpoints.
    u = edge_index[0, train_idx]
    v = edge_index[1, train_idx]
    centers_u = u
    centers_v = v

    print(f"[PT] {name}: building ND for {train_idx.numel()} train edges (x2 endpoints)")
    nd_u = build_nd_subgraph_indices(
        edge_index, N, centers_u,
        use_hop=use_hop, sample_size=sample_size, seed=seed,
    )
    nd_v = build_nd_subgraph_indices(
        edge_index, N, centers_v,
        use_hop=use_hop, sample_size=sample_size, seed=seed + 1,
    )
    y_rel = edge_types[train_idx]

    return dict(
        name=name, task_type="lp",
        feat=feat.to(device),
        y=y_rel.to(device),
        train_idx=torch.arange(train_idx.numel()),  # 0..n_edges-1 for slicing nd_u/nd_v
        nd_u=nd_u, nd_v=nd_v,
        label_names=rel_names,
    )


def _build_batch_lp(bundle, sl, tokenizer, max_text_len):
    feat = bundle["feat"]
    nd_u = bundle["nd_u"][sl]
    nd_v = bundle["nd_v"][sl]
    y_rel = bundle["y"][sl]
    rel_names = bundle["label_names"]

    tok_u, mask_u = lookup_nd_features(nd_u.to(feat.device), feat)
    tok_v, mask_v = lookup_nd_features(nd_v.to(feat.device), feat)
    tok = torch.cat([tok_u, tok_v], dim=1)
    graph_mask = torch.cat([mask_u, mask_v], dim=1)

    prompt = (
        f"Task: {bundle['name']} relation prediction. "
        f"Given two entity subgraphs (entity A followed by entity B), "
        f"predict the relation between them. Answer: "
    )
    prompts = [prompt] * len(sl)
    answers = [rel_names[int(y_rel[b].item())] for b in range(len(sl))]
    input_ids, targets, text_mask = _pack_text_batch(tokenizer, prompts, answers, max_text_len)

    return dict(
        tok=tok, graph_mask=graph_mask,
        input_ids=input_ids.to(feat.device),
        targets=targets.to(feat.device),
        text_mask=text_mask.to(feat.device),
    )


# ---------------------------------------------------------------------------
# GC bundle (chem datasets)
# ---------------------------------------------------------------------------


def _prepare_gc(name, device, use_hop, sample_size, seed, max_atoms=64, max_molecules=4000):
    """Load a chem dataset as a list of per-molecule token sequences.

    Pragmatic simplifications:
    - Cap at `max_molecules` random molecules (PCBA / blpre have millions
      of edges; full-scale pretrain is another week of work).
    - Truncate molecules longer than `max_atoms` atoms.
    - Use task 0 of the multi-task target as the binary supervision.
    """
    data, slices = load_tag(name)
    if slices is None or "x" not in slices:
        raise RuntimeError(f"{name}: chem dataset missing per-graph slices")

    x_slices = slices["x"].long()          # [num_graphs+1]
    ei_slices = slices["edge_index"].long() if "edge_index" in slices else None

    num_graphs = x_slices.numel() - 1
    atom_vocab_feat = data.node_text_feat.float()  # [V, 768]
    x_idx = data.x.squeeze(-1).long() if data.x.ndim == 2 else data.x.long()

    # multi-task binary labels, shape [num_graphs * num_tasks] or [num_graphs, num_tasks]
    y = data.y
    if y.ndim == 1:
        # flat; infer per-graph tasks from total/num_graphs
        n_tasks = y.numel() // num_graphs
        y = y.view(num_graphs, n_tasks)
    y = y.long()

    rng = np.random.default_rng(seed)
    take = min(max_molecules, num_graphs)
    gids = rng.choice(num_graphs, size=take, replace=False)

    # Build per-molecule token sequence table: we store ONE LongTensor
    # [take, max_atoms] with atom indices into the vocab, padded with -1.
    tokens = torch.full((take, max_atoms), -1, dtype=torch.long)
    per_mol_y = torch.zeros(take, dtype=torch.long)
    n_task = y.size(1)
    t_sel = 0  # fixed task 0 for all molecules
    for i, g in enumerate(gids):
        start, end = int(x_slices[g].item()), int(x_slices[g + 1].item())
        atoms = x_idx[start:end]
        if atoms.numel() > max_atoms:
            atoms = atoms[:max_atoms]
        tokens[i, :atoms.numel()] = atoms
        label = y[g, t_sel]
        per_mol_y[i] = label if label >= 0 else 0   # ignore nan flag → default no

    # Expand token-index table into [take, max_atoms, 768] when we build batches;
    # store only the index table here to save memory.
    return dict(
        name=name, task_type="gc",
        atom_vocab_feat=atom_vocab_feat.to(device),
        tokens=tokens,
        train_idx=torch.arange(take),
        y=per_mol_y.to(device),
        label_names=["no", "yes"],
        max_atoms=max_atoms,
    )


def _build_batch_gc(bundle, sl, tokenizer, max_text_len):
    toks = bundle["tokens"][sl]              # [b, L] index-or-minus1
    y = bundle["y"][sl]
    atom_feat = bundle["atom_vocab_feat"]    # [V, 768]
    graph_mask = (toks != -1).long()
    safe = toks.clamp(min=0).to(atom_feat.device)
    tok = atom_feat[safe] * graph_mask.unsqueeze(-1).to(atom_feat.dtype).to(atom_feat.device)

    prompt = (
        f"Task: {bundle['name']} molecular property prediction. "
        f"Given the atoms of the molecule, predict whether the property is active. "
        f"Answer: "
    )
    prompts = [prompt] * len(sl)
    answers = [bundle["label_names"][int(y[b].item())] for b in range(len(sl))]
    input_ids, targets, text_mask = _pack_text_batch(tokenizer, prompts, answers, max_text_len)

    return dict(
        tok=tok, graph_mask=graph_mask.to(atom_feat.device),
        input_ids=input_ids.to(atom_feat.device),
        targets=targets.to(atom_feat.device),
        text_mask=text_mask.to(atom_feat.device),
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def prepare_dataset(name, device, use_hop, sample_size, seed,
                    per_dataset_cap=None, gc_max_molecules=4000):
    task = DATASETS.get(name, {}).get("task", "node")
    if task == "node":
        return _prepare_nc(name, device, use_hop, sample_size, seed,
                           cap=per_dataset_cap)
    if task == "link":
        return _prepare_lp(name, device, use_hop, sample_size, seed,
                           cap=per_dataset_cap)
    if task == "graph":
        mm = min(gc_max_molecules, per_dataset_cap) if per_dataset_cap else gc_max_molecules
        return _prepare_gc(name, device, use_hop, sample_size, seed,
                           max_molecules=mm)
    raise ValueError(f"{name}: unknown task={task}")


def build_batch_indices(bundle, sl, tokenizer, use_hop, sample_size, max_text_len):
    t = bundle["task_type"]
    if t == "nc":
        return _build_batch_nc(bundle, sl, tokenizer, max_text_len)
    if t == "lp":
        return _build_batch_lp(bundle, sl, tokenizer, max_text_len)
    if t == "gc":
        return _build_batch_gc(bundle, sl, tokenizer, max_text_len)
    raise ValueError(f"unknown task_type={t}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def run_pretrain(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[PT] device={device} llm={args.llm_name}")

    cfg = LlagaConfig(
        llm_name_or_path=args.llm_name,
        mm_hidden_size=768,
        projector_type=args.projector_type,
        use_hop=args.use_hop,
        sample_size=args.sample_size,
        freeze_llm=True,
        llm_dtype=args.llm_dtype,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_impl,
    )

    from transformers import AutoTokenizer
    encoder = LlagaNDEncoder(cfg).to(device)
    llm = encoder.llm
    projector = encoder.mm_projector
    embed_tokens = llm.get_input_embeddings()

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name, cache_dir=cfg.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[PT] loading datasets: {args.datasets} per_cap={args.per_dataset_cap}")
    bundles = [
        prepare_dataset(name, device, args.use_hop, args.sample_size,
                        seed=0,
                        per_dataset_cap=args.per_dataset_cap,
                        gc_max_molecules=args.gc_max_molecules)
        for name in args.datasets
    ]
    for b in bundles:
        print(f"[PT]   {b['name']} ({b['task_type']}): "
              f"n_train={b['train_idx'].numel()} classes={len(b['label_names'])}")

    optim = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.wd)
    projector.train()
    llm.eval()
    llm_dtype = next(llm.parameters()).dtype

    def _llm_forward(graph_embeds, text_ids, text_mask, targets):
        text_embeds = embed_tokens(text_ids).to(graph_embeds.dtype)
        inp = torch.cat([graph_embeds, text_embeds], dim=1)
        B, L = graph_embeds.size(0), graph_embeds.size(1)
        T = text_ids.size(1)
        attn = torch.cat(
            [torch.ones(B, L, device=inp.device, dtype=torch.long), text_mask], dim=1,
        )
        out = llm(inputs_embeds=inp, attention_mask=attn, use_cache=False)
        logits = out.logits
        full_targets = torch.full((B, L + T), IGNORE_INDEX, device=inp.device, dtype=torch.long)
        full_targets[:, L:] = targets
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = full_targets[:, 1:].contiguous()
        # Guard against batches with zero supervised tokens (would produce NaN).
        if (shift_labels != IGNORE_INDEX).sum().item() == 0:
            return torch.zeros((), device=inp.device, dtype=shift_logits.dtype)
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    # Flat sampling pool across all bundles.
    index_pool: List[Tuple[int, int]] = []
    for bi, bundle in enumerate(bundles):
        for li in range(bundle["train_idx"].numel()):
            index_pool.append((bi, li))
    print(f"[PT] total train samples: {len(index_pool)}")

    rng = np.random.default_rng(0)
    step = 0
    t0 = time.time()
    losses = []

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(index_pool))
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, len(order), args.batch_size):
            end = min(start + args.batch_size, len(order))
            batch = [index_pool[i] for i in order[start:end]]
            by_ds: Dict[int, List[int]] = {}
            for bi, li in batch:
                by_ds.setdefault(bi, []).append(li)

            optim.zero_grad()
            total = 0.0
            for bi, locs in by_ds.items():
                bundle = bundles[bi]
                sl = torch.tensor(locs, dtype=torch.long)
                packed = build_batch_indices(
                    bundle, sl, tokenizer, args.use_hop, args.sample_size, args.max_text_len,
                )
                graph_embeds = projector(packed["tok"]).to(llm_dtype)
                graph_embeds = graph_embeds * packed["graph_mask"].unsqueeze(-1).to(graph_embeds.dtype)
                loss_bi = _llm_forward(
                    graph_embeds,
                    packed["input_ids"], packed["text_mask"], packed["targets"],
                )
                weight = len(locs) / len(batch)
                (loss_bi * weight).backward()
                total += float(loss_bi.item()) * weight

            optim.step()
            epoch_loss += total
            n_steps += 1
            step += 1

            if step % args.log_every == 0:
                dt = time.time() - t0
                ds_seen = ",".join(sorted({bundles[bi]["name"] for bi, _ in batch}))
                print(f"[PT] step={step} epoch={epoch} loss={total:.4f} "
                      f"ds_in_batch={ds_seen} elapsed={dt:.1f}s")

            if args.max_steps > 0 and step >= args.max_steps:
                break

        avg = epoch_loss / max(n_steps, 1)
        losses.append(avg)
        print(f"[PT] epoch={epoch} avg_loss={avg:.4f}")
        if args.max_steps > 0 and step >= args.max_steps:
            break

    os.makedirs(args.ckpt_dir, exist_ok=True)
    tag = args.tag
    pt_path = osp.join(args.ckpt_dir, f"projector_{tag}.pt")
    meta_path = osp.join(args.ckpt_dir, f"projector_{tag}.json")
    torch.save(projector.state_dict(), pt_path)
    meta = dict(
        llm_name=args.llm_name,
        mm_hidden_size=768,
        projector_type=args.projector_type,
        use_hop=args.use_hop,
        sample_size=args.sample_size,
        datasets=args.datasets,
        tasks={b["name"]: b["task_type"] for b in bundles},
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr, wd=args.wd,
        final_losses=losses,
        elapsed_seconds=time.time() - t0,
    )
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[PT] saved projector -> {pt_path}")
    print(f"[PT] saved meta      -> {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cora", "pubmed"],
                        help="Datasets to joint-pretrain on (GFT-9 supported)")
    parser.add_argument("--llm_name", default="lmsys/vicuna-7b-v1.5-16k")
    parser.add_argument("--llm_dtype", default="bfloat16")
    parser.add_argument("--attn_impl", default="eager")
    parser.add_argument("--projector_type", default="2-layer-mlp")
    parser.add_argument("--use_hop", type=int, default=2)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_text_len", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--ckpt_dir", default="ckpts/LLaGA")
    parser.add_argument("--tag", default="gft9")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--gc_max_molecules", type=int, default=4000,
                        help="Cap chem-dataset per-graph samples for memory")
    parser.add_argument("--per_dataset_cap", type=int, default=None,
                        help="Max training samples per dataset (balances FB15K237 vs cora)")
    args = parser.parse_args()
    run_pretrain(args)


if __name__ == "__main__":
    main()
