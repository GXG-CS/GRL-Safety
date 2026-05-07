"""Training helpers for the isolated OFA-full prompt-graph runner."""

from __future__ import annotations

import os
import os.path as osp
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from sgb.models.ofa.ofa_train_utils import (
    build_degree_ood_split,
    compute_node_degree,
    indices_to_mask,
    normalize_masks,
    step_imbalance_mask,
)
from sgb.models.gfm.ofa.ofa_bridge import build_model, make_csr_adj, prepare_base_graph


DEFAULT_TEXT_MODEL = "sentence-transformers/multi-qa-distilbert-cos-v1"


def default_prompt_text(dataset: str) -> str:
    citation = {"cora", "citeseer", "pubmed", "dblp", "arxiv", "arxiv23", "arxivyear"}
    ecommerce = {
        "amazonratings",
        "bookhis",
        "bookchild",
        "sportsfit",
        "elephoto",
        "elecomp",
        "products",
    }
    if dataset.lower() in citation:
        return "prompt node. node classification on the paper's category"
    if dataset.lower() in ecommerce:
        return "prompt node. node classification on the product category"
    if dataset.lower() == "wikics":
        return "prompt node. node classification on the wikipedia page category"
    if dataset.lower() == "tolokers":
        return "prompt node. node classification on the user label"
    return f"prompt node. node classification on the {dataset} graph"


def default_prompt_edge_text(dataset: str) -> str:
    if dataset.lower() in {"cora", "citeseer", "dblp"}:
        return "prompt edge"
    return "prompt edge."


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    emb = torch.sum(last_hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-10)
    return F.normalize(emb, p=2, dim=1)


def encode_prompt_texts(
    texts: list[str],
    *,
    model_cache_dir: Optional[str],
    device: torch.device,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TEXT_MODEL, cache_dir=model_cache_dir)
    model = AutoModel.from_pretrained(DEFAULT_TEXT_MODEL, cache_dir=model_cache_dir)
    model.to(device)
    model.eval()
    with torch.no_grad():
        batch = tokenizer(texts, padding="longest", truncation=True, max_length=500, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, output_hidden_states=True, return_dict=True)["hidden_states"][-1]
        emb = _mean_pool(out, batch["attention_mask"]).cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return emb


def get_prompt_features(
    data,
    *,
    dataset: str,
    project_root: str,
    model_cache_dir: Optional[str],
    device: torch.device,
):
    """Return NOI node and prompt-edge text embeddings.

    TAG registry data currently preserves class-node embeddings but often drops
    NOI/prompt-edge fields.  When absent, encode the OFA task templates with
    the same ST model used by the TAG cache and store a tiny local cache.
    """
    d = int(data.node_text_feat.size(1))
    noi = getattr(data, "noi_node_text_feat", None)
    pedge = getattr(data, "prompt_edge_text_feat", None)
    if noi is not None and pedge is not None:
        return noi[:1].float().to(device), pedge[:1].float().to(device)

    cache_dir = osp.join(project_root, "experiments", "ofa", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = osp.join(cache_dir, f"{dataset}_prompt_ST.pt")
    texts = [default_prompt_text(dataset), default_prompt_edge_text(dataset)]
    emb = None
    if osp.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("texts") == texts:
            emb = payload["emb"]
    if emb is None:
        enc_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        emb = encode_prompt_texts(texts, model_cache_dir=model_cache_dir, device=enc_device)
        torch.save({"texts": texts, "emb": emb}, cache_path)
    if emb.size(1) != d:
        raise ValueError(f"prompt embedding width {emb.size(1)} != TAG node width {d}")
    return emb[0:1].float().to(device), emb[1:2].float().to(device)


def mask_to_idx(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if mask.dtype == torch.long:
        return mask.to(device)
    return mask.bool().nonzero(as_tuple=False).view(-1).to(device)


def iter_chunks(idx: torch.Tensor, batch_size: int, shuffle: bool = False):
    if idx.numel() == 0:
        return
    if shuffle:
        idx = idx[torch.randperm(idx.numel(), device=idx.device)]
    if batch_size <= 0:
        yield idx
        return
    for start in range(0, idx.numel(), batch_size):
        yield idx[start : start + batch_size]


def maybe_cap_train(idx: torch.Tensor, labels: torch.Tensor, max_train_query: int, seed: int) -> torch.Tensor:
    if max_train_query <= 0 or idx.numel() <= max_train_query:
        return idx
    rng = np.random.RandomState(seed)
    y = labels[idx].detach().cpu().numpy()
    kept = []
    total = int(idx.numel())
    for c in sorted(set(y.tolist())):
        cidx = idx[torch.as_tensor(y == c, device=idx.device)]
        n_keep = max(1, int(round(cidx.numel() * max_train_query / total)))
        if cidx.numel() <= n_keep:
            kept.append(cidx)
        else:
            sel = rng.choice(cidx.numel(), size=n_keep, replace=False)
            kept.append(cidx[torch.as_tensor(sel, device=idx.device, dtype=torch.long)])
    out = torch.cat(kept) if kept else idx[:max_train_query]
    return out[:max_train_query] if out.numel() > max_train_query else out


def graph_view(g, *, x=None, edge_index=None, edge_attr=None, edge_type=None):
    out_edge_index = g.edge_index if edge_index is None else edge_index
    out_edge_attr = g.edge_attr if edge_attr is None else edge_attr
    if edge_index is None and hasattr(g, "adj"):
        adj = g.adj
    else:
        adj = make_csr_adj(out_edge_index, g.x.size(0))
    original_edge_feature = (
        out_edge_attr[:1].contiguous()
        if out_edge_attr is not None and out_edge_attr.numel() > 0
        else getattr(g, "original_edge_feature", None)
    )
    return SimpleNamespace(
        x=g.x if x is None else x,
        edge_index=out_edge_index,
        edge_attr=out_edge_attr,
        edge_type=g.edge_type if edge_type is None else edge_type,
        adj=adj,
        original_edge_feature=original_edge_feature,
        prompt_graph_mode=getattr(g, "prompt_graph_mode", "subgraph"),
        subgraph_hops=getattr(g, "subgraph_hops", 2),
        max_nodes_per_hop=getattr(g, "max_nodes_per_hop", 100),
        subgraph_seed=getattr(g, "subgraph_seed", 0),
    )


def prompt_logits(model, g, cls_emb, noi_emb, prompt_edge_emb, query_idx, *, batch_size: int, grad: bool = False):
    outs = []
    was_training = model.training
    if not grad:
        model.eval()
    ctx = torch.enable_grad() if grad else torch.no_grad()
    try:
        with ctx:
            for q in iter_chunks(query_idx, batch_size, shuffle=False):
                outs.append(model(g, cls_emb, noi_emb, q, prompt_edge_emb))
    finally:
        if not grad and was_training:
            model.train()
    if outs:
        return torch.cat(outs, dim=0)
    return torch.empty(0, cls_emb.size(0), device=cls_emb.device)


@torch.no_grad()
def accuracy(model, g, cls_emb, noi_emb, prompt_edge_emb, labels, mask_or_idx, *, batch_size: int) -> float:
    idx = mask_or_idx if mask_or_idx.dtype == torch.long else mask_to_idx(mask_or_idx, labels.device)
    if idx.numel() == 0:
        return float("nan")
    logits = prompt_logits(model, g, cls_emb, noi_emb, prompt_edge_emb, idx, batch_size=batch_size)
    pred = logits.argmax(dim=-1)
    return float((pred == labels[idx]).float().mean().item())


def per_class_recall(logits: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, num_classes: int) -> np.ndarray:
    pred = logits.argmax(dim=-1).detach().cpu().numpy()
    true = labels[idx].detach().cpu().numpy()
    out = np.full(num_classes, np.nan)
    for c in range(num_classes):
        sel = true == c
        if sel.sum() > 0:
            out[c] = float((pred[sel] == c).mean())
    return out


@torch.no_grad()
def balanced_accuracy(model, g, cls_emb, noi_emb, prompt_edge_emb, labels, mask_or_idx, *, batch_size: int) -> float:
    idx = mask_or_idx if mask_or_idx.dtype == torch.long else mask_to_idx(mask_or_idx, labels.device)
    if idx.numel() == 0:
        return float("nan")
    logits = prompt_logits(model, g, cls_emb, noi_emb, prompt_edge_emb, idx, batch_size=batch_size)
    rec = per_class_recall(logits, labels, idx, int(cls_emb.size(0)))
    return float(np.nanmean(rec))


def link_prediction_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """OFA-style class-node link BCE for single-label node classification."""
    target = F.one_hot(labels, num_classes=num_classes).to(dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target)


def train_prompt_model(data, train_mask, val_mask, *, seed: int, device: torch.device, args, val_metric: str = "acc"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    g, cls_emb = prepare_base_graph(data, device=device)
    g.prompt_graph_mode = args.prompt_graph_mode
    g.subgraph_hops = args.subgraph_hops
    g.max_nodes_per_hop = args.max_nodes_per_hop
    g.subgraph_seed = seed
    noi_emb, prompt_edge_emb = get_prompt_features(
        data,
        dataset=args.dataset,
        project_root=args.project_root,
        model_cache_dir=args.model_cache_dir,
        device=device,
    )
    labels = data.y.squeeze().long().to(device)
    train_idx = maybe_cap_train(mask_to_idx(train_mask, device), labels, args.max_train_query, seed)
    val_idx = mask_to_idx(val_mask, device)

    model = build_model(
        llm_name=args.llm_name,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_rels=args.num_rels,
        dropout=args.dropout,
        jk=args.jk,
        pretrained_encoder=args.pretrained_encoder,
    ).to(device)
    if args.freeze_encoder:
        for module in (model.encoder, model.llm_proj):
            for p in module.parameters():
                p.requires_grad = False
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=max(1, int(args.epochs)))

    best_val = -1.0
    best_state = None
    bad = 0
    for _ in range(args.epochs):
        model.train()
        for q in iter_chunks(train_idx, args.train_query_batch_size, shuffle=True):
            opt.zero_grad()
            logits = model(g, cls_emb, noi_emb, q, prompt_edge_emb)
            loss = link_prediction_loss(logits, labels[q], int(cls_emb.size(0)))
            loss.backward()
            opt.step()
        scheduler.step()

        if val_metric == "bacc":
            score = balanced_accuracy(
                model, g, cls_emb, noi_emb, prompt_edge_emb, labels, val_idx,
                batch_size=args.eval_query_batch_size,
            )
        else:
            score = accuracy(
                model, g, cls_emb, noi_emb, prompt_edge_emb, labels, val_idx,
                batch_size=args.eval_query_batch_size,
            )
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, g, cls_emb, noi_emb, prompt_edge_emb, labels
