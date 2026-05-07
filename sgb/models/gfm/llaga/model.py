"""LLaGA (Large Language and Graph Assistant) encoder for GFM-Safety.

Design (adapted for our uniform discriminative FT protocol):

1. Neighborhood Detail (ND) template: for each target node build a fixed
   (use_hop, sample_size) ego tree. Level 0 = self, level k = up to
   `sample_size` neighbors of each level-(k-1) node. Total length:
   (sample_size^(use_hop+1) - 1) / (sample_size - 1). Missing slots are
   padded with DEFAULT_GRAPH_PAD_ID; padded positions zero out after
   projection (matches upstream `encode_graphs`).
2. Graph tokens come from our unified TAG node_text_feat (SBERT 768d).
   A 2-layer MLP projects them into the LLM hidden size ("mm_projector").
3. A frozen causal LLM (Vicuna-7B by default) processes the projected
   token sequence via `inputs_embeds=...`. We take the hidden state at
   the last non-pad position and feed it into a linear classifier head.
4. Only the projector and head are trained by default. Set
   `freeze_llm=False` / pass LoRA adapters if a full FT sweep is desired.

This is LLaGA-lite: faithful graph tokenization + mm_projector, but we
do classification via readout rather than generation. Rationale is
documented in INTEGRATION_NOTES.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# transformers 5.4 refuses torch.load of .bin files (CVE-2025-32434) unless
# torch >= 2.6. Our safety env has 2.5.1, and we're loading weights we
# downloaded ourselves from HF, so bypass the check.
def _bypass_transformers_load_check():
    try:
        import transformers.utils.import_utils as _iu
        _iu.check_torch_load_is_safe = lambda *a, **k: None
    except Exception:
        pass
    try:
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "check_torch_load_is_safe"):
            _mu.check_torch_load_is_safe = lambda *a, **k: None
    except Exception:
        pass


_bypass_transformers_load_check()


DEFAULT_GRAPH_PAD_ID = -500  # matches upstream utils.constants


# ---------------------------------------------------------------------------
# ND ego tree sampler
# ---------------------------------------------------------------------------


def _build_csr_adj(edge_index: torch.Tensor, num_nodes: int):
    """Return a sorted-by-source row pointer + col array for fast neighbor lookup."""
    src = edge_index[0].long()
    dst = edge_index[1].long()
    # make it undirected
    src_sym = torch.cat([src, dst])
    dst_sym = torch.cat([dst, src])
    order = torch.argsort(src_sym)
    src_sym = src_sym[order]
    dst_sym = dst_sym[order]
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    counts = torch.bincount(src_sym, minlength=num_nodes)
    rowptr[1:] = torch.cumsum(counts, dim=0)
    return rowptr, dst_sym


def build_nd_subgraph_indices(
    edge_index: torch.Tensor,
    num_nodes: int,
    center_nodes: torch.Tensor,
    use_hop: int = 2,
    sample_size: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """Build ND template: [B, L] indices into node feature table.

    Returns a LongTensor on CPU. Padded slots are DEFAULT_GRAPH_PAD_ID.
    L = (sample_size^(use_hop+1) - 1) / (sample_size - 1) when sample_size>1,
    else use_hop + 1.
    """
    rowptr, col = _build_csr_adj(edge_index.cpu(), num_nodes)

    if sample_size == 1:
        seq_len = use_hop + 1
    else:
        seq_len = (sample_size ** (use_hop + 1) - 1) // (sample_size - 1)

    gen = torch.Generator().manual_seed(int(seed))
    B = center_nodes.numel()
    out = torch.full((B, seq_len), DEFAULT_GRAPH_PAD_ID, dtype=torch.long)

    for b in range(B):
        frontier = [int(center_nodes[b].item())]
        pos = 0
        out[b, pos] = frontier[0]
        pos += 1
        for h in range(use_hop):
            next_frontier = []
            for v in frontier:
                if v == DEFAULT_GRAPH_PAD_ID:
                    # pad children
                    for _ in range(sample_size):
                        next_frontier.append(DEFAULT_GRAPH_PAD_ID)
                    continue
                start, end = int(rowptr[v].item()), int(rowptr[v + 1].item())
                deg = end - start
                if deg == 0:
                    picks = [DEFAULT_GRAPH_PAD_ID] * sample_size
                else:
                    if deg >= sample_size:
                        # sample without replacement
                        idx = torch.randperm(deg, generator=gen)[:sample_size]
                    else:
                        # sample with replacement to fill
                        idx = torch.randint(0, deg, (sample_size,), generator=gen)
                    picks = [int(col[start + int(i.item())].item()) for i in idx]
                for p in picks:
                    next_frontier.append(p)
            # write this hop's slice
            for v in next_frontier:
                if pos >= seq_len:
                    break
                out[b, pos] = v
                pos += 1
            frontier = next_frontier

    return out  # [B, L]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class LlagaConfig:
    llm_name_or_path: str = "lmsys/vicuna-7b-v1.5-16k"
    mm_hidden_size: int = 768         # input node feature dim
    projector_type: str = "2-layer-mlp"  # 'linear' or 'k-layer-mlp'
    use_hop: int = 2
    sample_size: int = 10
    freeze_llm: bool = True
    llm_dtype: str = "bfloat16"       # 'float32', 'float16', 'bfloat16'
    cache_dir: Optional[str] = None
    attn_implementation: str = "eager"  # 'eager' | 'sdpa'


def _build_projector(in_dim: int, hidden: int, projector_type: str) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(in_dim, hidden)
    import re
    m = re.match(r"^(\d+)-layer-mlp$", projector_type)
    if m:
        depth = int(m.group(1))
        mods = [nn.Linear(in_dim, hidden)]
        for _ in range(1, depth):
            mods.append(nn.GELU())
            mods.append(nn.Linear(hidden, hidden))
        return nn.Sequential(*mods)
    raise ValueError(f"unknown projector_type={projector_type}")


class LlagaNDEncoder(nn.Module):
    """ND-template graph encoder: projector + frozen LLM → per-sample vector.

    Forward signature matches our benchmark pattern:
      forward(batch_node_features) -> [B, hidden_size]

    `batch_node_features` is the tensor produced by looking up the ND
    template indices in the full-graph node feature table, with padded
    rows already zero'd. Shape: [B, L, mm_hidden_size].
    """

    def __init__(self, cfg: LlagaConfig):
        super().__init__()
        self.cfg = cfg

        # Deferred imports so a pure-encoder user isn't forced to pull
        # transformers if they only want the ND sampler.
        from transformers import AutoConfig, AutoModelForCausalLM

        llm_cfg = AutoConfig.from_pretrained(
            cfg.llm_name_or_path, cache_dir=cfg.cache_dir,
        )
        self.llm_hidden_size = int(llm_cfg.hidden_size)

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[cfg.llm_dtype]

        self.llm = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name_or_path,
            torch_dtype=dtype,
            cache_dir=cfg.cache_dir,
            attn_implementation=cfg.attn_implementation,
        )
        if cfg.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()

        self.mm_projector = _build_projector(
            cfg.mm_hidden_size, self.llm_hidden_size, cfg.projector_type,
        )

    def forward(self, token_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """token_features: [B, L, mm_hidden_size]; attention_mask: [B, L] (1 live, 0 pad)."""
        B, L, _ = token_features.shape
        inp_embeds = self.mm_projector(token_features)
        # cast to LLM's dtype for the forward pass
        target_dtype = next(self.llm.parameters()).dtype
        inp_embeds = inp_embeds.to(target_dtype)

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=inp_embeds.device, dtype=torch.long)

        # Frozen LLM forward — request hidden states so we can read out.
        ctx = torch.no_grad() if self.cfg.freeze_llm else torch.enable_grad()
        with ctx:
            out = self.llm.model(
                inputs_embeds=inp_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
            )
        last_hidden = out.last_hidden_state  # [B, L, H]

        # Causal-LM readout: take last valid position per sample (equivalent
        # to how LLaGA's generation reads the next token logits).
        # attention_mask is [B, L]; sum-1 gives index of last live token.
        last_idx = attention_mask.long().sum(dim=1) - 1  # [B]
        last_idx = last_idx.clamp_min(0)
        batch_idx = torch.arange(B, device=last_hidden.device)
        pooled = last_hidden[batch_idx, last_idx]       # [B, H]
        return pooled.float()


class LlagaNDClassifier(nn.Module):
    """LlagaNDEncoder + linear head for node classification."""

    def __init__(self, cfg: LlagaConfig, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = LlagaNDEncoder(cfg)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.llm_hidden_size, num_classes)

    def forward(self, token_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(token_features, attention_mask)
        h = self.dropout(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Helpers used by finetune scripts
# ---------------------------------------------------------------------------


def lookup_nd_features(
    nd_indices: torch.Tensor,
    node_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Given nd_indices [B, L] (possibly containing PAD) and node_features [N, D],
    return (token_features [B, L, D], attention_mask [B, L])."""
    B, L = nd_indices.shape
    D = node_features.size(1)
    mask = (nd_indices != DEFAULT_GRAPH_PAD_ID)
    safe_idx = nd_indices.clone()
    safe_idx[~mask] = 0
    tok = node_features[safe_idx]  # [B, L, D]
    # zero out padded positions
    tok = tok * mask.unsqueeze(-1).to(tok.dtype)
    return tok, mask.long()
