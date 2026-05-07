"""OFA bridge: use OFA's GNN encoder on GFM-Safety's unified TAG data.

Design rationale
----------------
OFA's full design = prompt-graph construction + class-nodes-as-prompts +
task-constructor + lightning trainer. Replicating that faithfully against
GFM-Safety's tag_registry is a multi-week project. This bridge instead keeps
OFA's *core* contributions and bypasses the infrastructure scaffolding:

  (a) OFA's RGCN encoder (`PyGRGCNEdge`) acts on the graph.
  (b) OFA's LM-projection (`llm_proj`) maps SBERT-768 node and class text
      features into a shared d-dim space so class text becomes the class head.
  (c) Class logits are produced by matching each node's GNN embedding against
      the projected class-text embeddings (text-as-classifier, which is the
      signature inductive bias of OFA).

What this bridge deliberately skips (can be upgraded later):
  - Prompt graph with NOI/class/feature node types and the f2n/n2c/... edges.
  - UnifiedTaskConstructor / lightning datamodule / multi-task joint pretrain.
  - Few-shot / ICL path.

The outcome is a faithful "OFA-as-encoder" baseline that can run the 8 safety
dimensions using GFT's fine-tune template.
"""

from __future__ import annotations

import os
import os.path as osp
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = osp.dirname(osp.abspath(__file__))
_OFA_DIR = osp.join(_HERE, "OFA")
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))

# OFA imports require sys.path entry because models/gp modules live under OFA/.
if _OFA_DIR not in sys.path:
    sys.path.insert(0, _OFA_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.model import PyGRGCNEdge, LLM_DIM_DICT  # noqa: E402


class OFAEncoderHead(nn.Module):
    """OFA-lite: RGCN encoder + LM-projected-class-text classifier.

    Forward:
        g.x           : [N, llm_dim] node SBERT embedding (= node_text_feat)
        g.edge_index  : [2, E]
        g.edge_type   : [E] int edge-type indices (all zeros for one-relation)
        g.edge_attr   : [E, llm_dim] edge SBERT embedding
        class_text_emb: [C, llm_dim] class SBERT embedding (= class_node_text_feat)

    Returns logits : [N, C]
    """

    def __init__(
        self,
        llm_name: str = "ST",
        emb_dim: int = 768,
        num_layers: int = 6,
        num_rels: int = 5,  # OFA pretrain uses 5 relation types (f2n/n2f/n2c/c2n + self)
        dropout: float = 0.0,
        jk: str = "last",
    ) -> None:
        super().__init__()
        assert llm_name in LLM_DIM_DICT, f"unknown llm_name {llm_name}"
        llm_dim = LLM_DIM_DICT[llm_name]
        self.llm_name = llm_name
        self.emb_dim = emb_dim
        self.llm_proj = nn.Linear(llm_dim, emb_dim)
        self.encoder = PyGRGCNEdge(
            num_layers=num_layers,
            num_rels=num_rels,
            inp_dim=emb_dim,
            out_dim=emb_dim,
            drop_ratio=dropout,
            JK=jk,
            batch_norm=True,
        )
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        # Init out_proj to identity so the pretrained encoder/llm_proj alignment
        # with class-text embeddings is preserved at step 0 (without this, a
        # random rotation scrambles cosine matching against class_proj and the
        # head has to relearn alignment from a tiny imbalanced labeled set).
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, g, class_text_emb: torch.Tensor) -> torch.Tensor:
        # Project LM features to the GNN hidden dim.
        x = self.llm_proj(g.x)
        edge_attr = self.llm_proj(g.edge_attr)

        # Carry projected features through PyGRGCNEdge (it reads g.x / g.edge_attr).
        # We reuse g but replace the tensors so the object shape is unchanged.
        g.x = x
        g.edge_attr = edge_attr
        node_emb = self.encoder(g)
        node_emb = self.out_proj(node_emb)

        # Class head: project class text embeddings through the same LM proj
        # (shared weights — matches OFA's "shared LM projection" design).
        class_proj = self.llm_proj(class_text_emb)

        # Cosine logits (temperature-free; works stably for fine-tune with CE).
        node_n = F.normalize(node_emb, dim=-1)
        class_n = F.normalize(class_proj, dim=-1)
        logits = node_n @ class_n.T * 10.0  # scale to give CE reasonable range
        return logits

    def forward_prompt(
        self,
        g,
        class_text_emb: torch.Tensor,
        query_indices: torch.Tensor,
    ) -> torch.Tensor:
        """OFA NC prompt-graph forward (graph-wide variant).

        Appends C class nodes to the original graph and adds bidirectional
        edges between every query node and every class node (n2c rel=2 +
        c2n rel=3 — OFA's two NC-specific relation slots). The RGCN then
        propagates label information from class nodes into query nodes
        across all layers of message passing — this is OFA's defining
        inductive bias for NC, distinct from a final-layer cosine match.

        Args:
            g: SimpleNamespace with x/edge_index/edge_type/edge_attr (raw,
               not yet llm_proj-projected; matches `prepare_ofa_input`).
            class_text_emb: [C, llm_dim]
            query_indices: [Q] long tensor of node indices in g whose class
               we want to predict (train batch or test set).

        Returns:
            logits: [Q, C]
        """
        device = g.x.device
        N = g.x.size(0)
        C = class_text_emb.size(0)
        Q = query_indices.size(0)
        llm_dim = g.x.size(1)

        # Concatenate node features + class text features.
        x_ext = torch.cat([g.x, class_text_emb.to(device)], dim=0)  # [N+C, llm_dim]
        cls_idx_ext = torch.arange(C, device=device) + N  # [C]

        # query → class (n2c rel=2)
        n2c_src = query_indices.repeat_interleave(C)  # [Q*C]
        n2c_dst = cls_idx_ext.repeat(Q)               # [Q*C]
        # class → query (c2n rel=3)
        c2n_src = cls_idx_ext.repeat(Q)
        c2n_dst = query_indices.repeat_interleave(C)

        new_edge_index = torch.cat([
            g.edge_index,
            torch.stack([n2c_src, n2c_dst], dim=0),
            torch.stack([c2n_src, c2n_dst], dim=0),
        ], dim=1)

        # Edge types: original=0 (f2n), n2c=2, c2n=3 (matches OFA's 5-rel slots).
        new_edge_type = torch.cat([
            torch.zeros(g.edge_index.size(1), dtype=torch.long, device=device),
            torch.full((Q * C,), 2, dtype=torch.long, device=device),
            torch.full((Q * C,), 3, dtype=torch.long, device=device),
        ])

        # Edge attrs: original original; new edges get zero edge_attr (OFA
        # doesn't use edge text for prompt edges in NC).
        new_edge_attr = torch.cat([
            g.edge_attr,
            torch.zeros(Q * C * 2, llm_dim, dtype=g.edge_attr.dtype, device=device),
        ], dim=0)

        # Project + forward through encoder.
        x_proj = self.llm_proj(x_ext)
        edge_attr_proj = self.llm_proj(new_edge_attr)

        from types import SimpleNamespace
        g_ext = SimpleNamespace(
            x=x_proj,
            edge_index=new_edge_index,
            edge_type=new_edge_type,
            edge_attr=edge_attr_proj,
        )
        z = self.encoder(g_ext)
        z = self.out_proj(z)

        # Read class node embeddings; cosine logits per query.
        z_class = z[cls_idx_ext]              # [C, emb_dim]
        z_query = z[query_indices]            # [Q, emb_dim]
        z_q_n = F.normalize(z_query, dim=-1)
        z_c_n = F.normalize(z_class, dim=-1)
        logits = z_q_n @ z_c_n.T * 10.0       # [Q, C]
        return logits


def build_model(
    llm_name: str = "ST",
    emb_dim: int = 768,
    num_layers: int = 6,
    num_rels: int = 5,  # OFA pretrain uses 5 relation types (f2n/n2f/n2c/c2n + self)
    dropout: float = 0.1,
    jk: str = "last",
    pretrained_encoder: Optional[str] = None,
) -> OFAEncoderHead:
    """Build an OFA-lite model; optionally init encoder+llm_proj from
    a file produced by `sgb/models/gfm/ofa/load_pretrained.py`.
    """
    model = OFAEncoderHead(
        llm_name=llm_name,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_rels=num_rels,
        dropout=dropout,
        jk=jk,
    )

    if pretrained_encoder is not None:
        assert osp.exists(pretrained_encoder), (
            f"pretrained_encoder file missing: {pretrained_encoder}"
        )
        payload = torch.load(pretrained_encoder, map_location="cpu", weights_only=False)
        # strict=False so any shape mismatch (e.g. num_rels differs) just skips.
        enc_missing = model.encoder.load_state_dict(payload["encoder"], strict=False)
        proj_missing = model.llm_proj.load_state_dict(payload["llm_proj"], strict=False)
        print(
            f"[ofa_bridge] pretrained loaded from {pretrained_encoder}  "
            f"encoder missing/unexpected={enc_missing}  llm_proj missing/unexpected={proj_missing}"
        )
    return model


def _ensure_edge_type(edge_index: torch.Tensor, num_rels: int = 1) -> torch.Tensor:
    """All-zero edge_type tensor for single-relation graphs.

    PyGRGCNEdge requires edge_type; for plain homogeneous TAG graphs we set
    all edges to relation 0 and num_rels=1.
    """
    return torch.zeros(edge_index.size(1), dtype=torch.long)


def prepare_ofa_input(
    data,
    device: Optional[torch.device] = None,
    num_rels: int = 5,  # OFA pretrain uses 5 relation types (f2n/n2f/n2c/c2n + self)
):
    """Fill in fields PyGRGCNEdge + OFAEncoderHead need.

    Expects a PyG `Data` object whose `node_text_feat` (N, 768) and
    `class_node_text_feat` (C, 768) were populated by tag_registry.load().
    Returns a fresh graph tensor container with the right fields and the
    class text embedding (C, 768).
    """
    assert getattr(data, "node_text_feat", None) is not None, (
        "OFA bridge expects node_text_feat on the data object — "
        "run through sgb.data.tag_registry.load()."
    )
    assert getattr(data, "class_node_text_feat", None) is not None, (
        "OFA bridge expects class_node_text_feat on the data object."
    )

    x = data.node_text_feat.float()
    edge_index = data.edge_index
    edge_type = _ensure_edge_type(edge_index, num_rels)

    # edge_attr: if tag_registry supplies a single per-relation edge embedding,
    # broadcast it over all edges. Else use zeros (same dim as x).
    etf = getattr(data, "edge_text_feat", None)
    if etf is None:
        edge_attr = torch.zeros(edge_index.size(1), x.size(1), dtype=x.dtype)
    else:
        etf = etf.float()
        if etf.dim() == 2 and etf.size(0) == 1:
            # Broadcast the single edge-type embedding to every edge.
            # Use expand (stride-0 view) instead of contiguous expand to
            # avoid materializing a fresh (E, 768) tensor — saves several
            # GB on large graphs (sportsfit/bookchild OOM on 40GB GPU).
            edge_attr = etf.expand(edge_index.size(1), -1)
        elif etf.dim() == 2 and etf.size(0) == edge_index.size(1):
            edge_attr = etf
        else:
            edge_attr = etf[0:1].expand(edge_index.size(1), -1)

    class_text_emb = data.class_node_text_feat.float()

    # Build a lightweight container (PyG Data also works; lean on plain Namespace).
    from types import SimpleNamespace
    g = SimpleNamespace(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        edge_attr=edge_attr,
    )

    if device is not None:
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        g.edge_type = g.edge_type.to(device)
        g.edge_attr = g.edge_attr.to(device)
        class_text_emb = class_text_emb.to(device)

    return g, class_text_emb
