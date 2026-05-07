"""OFA-full prompt graph bridge for GFM-Safety TAG node classification.

This module is intentionally isolated from ``sgb.models.ofa``.  It keeps the
existing OFA-lite bridge untouched and implements the paper-style prompt graph:
query node, NOI prompt node, one class node per class, typed prompt edges, and
class-node link logits trained with OFA's binary link-prediction objective.
"""

from __future__ import annotations

import importlib.machinery
import os.path as osp
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
_OFA_DIR = osp.join(_PROJECT_ROOT, "sgb", "models", "gfm", "ofa", "OFA")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _OFA_DIR not in sys.path:
    sys.path.insert(0, _OFA_DIR)


def _install_deepspeed_stub() -> None:
    """OFA imports a ZeRO helper at module import time; this path never uses it."""
    if "deepspeed.utils.zero_to_fp32" in sys.modules:
        return
    ds_mod = ModuleType("deepspeed")
    ds_utils = ModuleType("deepspeed.utils")
    ds_zero = ModuleType("deepspeed.utils.zero_to_fp32")
    ds_mod.__spec__ = importlib.machinery.ModuleSpec("deepspeed", loader=None, is_package=True)
    ds_mod.__path__ = []
    ds_utils.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils", loader=None, is_package=True)
    ds_utils.__path__ = []
    ds_zero.__spec__ = importlib.machinery.ModuleSpec("deepspeed.utils.zero_to_fp32", loader=None)

    def _unused_zero_checkpoint(*_args, **_kwargs):
        raise RuntimeError("DeepSpeed ZeRO checkpoint conversion is not used by OFA-full.")

    ds_zero.get_fp32_state_dict_from_zero_checkpoint = _unused_zero_checkpoint
    ds_utils.zero_to_fp32 = ds_zero
    ds_mod.utils = ds_utils
    sys.modules["deepspeed"] = ds_mod
    sys.modules["deepspeed.utils"] = ds_utils
    sys.modules["deepspeed.utils.zero_to_fp32"] = ds_zero


_install_deepspeed_stub()

from models.model import LLM_DIM_DICT, PyGRGCNEdge, SingleHeadAtt  # noqa: E402
from gp.nn.models.util_model import MLP  # noqa: E402


@dataclass(frozen=True)
class PromptRelations:
    """Relation ids used by the local OFA pretrain code.

    The upstream OFA e2e node prompt map is:
    original graph edge = 0, f2n/query-to-NOI = 1, n2c/NOI-to-class = 2,
    n2f/NOI-to-query = 3, c2n/class-to-NOI = 4.
    """

    ORIGINAL: int = 0
    QUERY_TO_NOI: int = 1
    NOI_TO_CLASS: int = 2
    NOI_TO_QUERY: int = 3
    CLASS_TO_NOI: int = 4


OFA_PROMPT_RELATIONS = PromptRelations()


def _as_2d_prompt_feature(feat: torch.Tensor, width: int, device: torch.device) -> torch.Tensor:
    feat = feat.to(device=device, dtype=torch.float32)
    if feat.dim() == 1:
        feat = feat.view(1, -1)
    if feat.size(0) != 1:
        feat = feat[:1]
    if feat.size(1) != width:
        raise ValueError(f"prompt feature width {feat.size(1)} != expected {width}")
    return feat


def _repeat_edge_feature(edge_feat: Optional[torch.Tensor], count: int, width: int, device: torch.device) -> torch.Tensor:
    if count == 0:
        return torch.empty(0, width, device=device)
    if edge_feat is None:
        return torch.zeros(count, width, device=device)
    edge_feat = _as_2d_prompt_feature(edge_feat, width, device)
    return edge_feat.expand(count, -1)


def make_csr_adj(edge_index: torch.Tensor, num_nodes: int):
    """Build the CSR adjacency used by OFA's fixed-hop subgraph sampler."""
    from scipy.sparse import csr_array

    ei = edge_index.detach().cpu()
    rows = ei[0].numpy()
    cols = ei[1].numpy()
    vals = np.ones(rows.shape[0], dtype=np.int8)
    return csr_array((vals, (rows, cols)), shape=(num_nodes, num_nodes))


def _sample_fixed_hop_neighbors(adj, root: int, hop: int, max_nodes_per_hop: int, seed: int) -> np.ndarray:
    """Deterministic variant of OFA's fixed-hop neighbor sampler."""
    visited = np.array([root], dtype=np.int64)
    fringe = np.array([root], dtype=np.int64)
    nodes = []
    for h in range(1, hop + 1):
        cand = adj[fringe].nonzero()[1].astype(np.int64)
        fringe = np.setdiff1d(cand, visited, assume_unique=False)
        visited = np.union1d(visited, fringe)
        if max_nodes_per_hop > 0 and len(fringe) > max_nodes_per_hop:
            rng = np.random.RandomState(seed + root * 1009 + h * 9173)
            fringe = np.sort(rng.choice(fringe, max_nodes_per_hop, replace=False))
        if len(fringe) == 0:
            break
        nodes.append(fringe)
    if not nodes:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(nodes).astype(np.int64)


def build_node_classification_prompt(
    base_g,
    class_text_emb: torch.Tensor,
    noi_text_emb: torch.Tensor,
    query_indices: torch.Tensor,
    prompt_edge_text_emb: Optional[torch.Tensor] = None,
    relations: PromptRelations = OFA_PROMPT_RELATIONS,
):
    """Append per-query NOI and class nodes to a base TAG graph.

    Class nodes are duplicated per query so batched queries cannot exchange
    information through shared prompt nodes.
    """
    device = base_g.x.device
    query_indices = query_indices.to(device=device, dtype=torch.long).view(-1)
    class_text_emb = class_text_emb.to(device=device, dtype=torch.float32)
    if class_text_emb.dim() != 2:
        raise ValueError("class_text_emb must have shape [num_classes, llm_dim]")

    q = int(query_indices.numel())
    c = int(class_text_emb.size(0))
    d = int(base_g.x.size(1))
    n = int(base_g.x.size(0))
    if q == 0:
        raise ValueError("query_indices is empty")
    if class_text_emb.size(1) != d:
        raise ValueError(f"class feature width {class_text_emb.size(1)} != node width {d}")
    noi_text_emb = _as_2d_prompt_feature(noi_text_emb, d, device)

    prompt_block = torch.cat([noi_text_emb, class_text_emb], dim=0)
    prompt_x = prompt_block.unsqueeze(0).expand(q, c + 1, d).reshape(q * (c + 1), d)
    x = torch.cat([base_g.x.to(device=device, dtype=torch.float32), prompt_x], dim=0)

    offsets = n + torch.arange(q, device=device, dtype=torch.long) * (c + 1)
    noi_idx = offsets
    class_offsets = torch.arange(c, device=device, dtype=torch.long).view(1, c)
    class_idx = offsets.view(q, 1) + 1 + class_offsets

    query_to_noi = torch.stack([query_indices, noi_idx], dim=0)
    noi_to_query = torch.stack([noi_idx, query_indices], dim=0)
    noi_to_class = torch.stack(
        [noi_idx.repeat_interleave(c), class_idx.reshape(-1)],
        dim=0,
    )
    class_to_noi = torch.stack(
        [class_idx.reshape(-1), noi_idx.repeat_interleave(c)],
        dim=0,
    )

    edge_index = torch.cat(
        [
            base_g.edge_index.to(device=device, dtype=torch.long),
            query_to_noi,
            noi_to_query,
            noi_to_class,
            class_to_noi,
        ],
        dim=1,
    )

    base_edge_count = int(base_g.edge_index.size(1))
    base_edge_type = getattr(base_g, "edge_type", None)
    if base_edge_type is None:
        base_edge_type = torch.full((base_edge_count,), relations.ORIGINAL, dtype=torch.long, device=device)
    else:
        base_edge_type = base_edge_type.to(device=device, dtype=torch.long)

    edge_type = torch.cat(
        [
            base_edge_type,
            torch.full((q,), relations.QUERY_TO_NOI, dtype=torch.long, device=device),
            torch.full((q,), relations.NOI_TO_QUERY, dtype=torch.long, device=device),
            torch.full((q * c,), relations.NOI_TO_CLASS, dtype=torch.long, device=device),
            torch.full((q * c,), relations.CLASS_TO_NOI, dtype=torch.long, device=device),
        ],
        dim=0,
    )

    base_edge_attr = getattr(base_g, "edge_attr", None)
    if base_edge_attr is None:
        base_edge_attr = getattr(base_g, "edge_text_feat", None)
    if base_edge_attr is None:
        base_edge_attr = torch.zeros((base_edge_count, d), dtype=torch.float32, device=device)
    else:
        base_edge_attr = base_edge_attr.to(device=device, dtype=torch.float32)
    prompt_edge_count = q * 2 + q * c * 2
    prompt_edge_attr = _repeat_edge_feature(prompt_edge_text_emb, prompt_edge_count, d, device)
    edge_attr = torch.cat([base_edge_attr, prompt_edge_attr], dim=0)

    return SimpleNamespace(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        query_node_index=query_indices,
        noi_node_index=noi_idx,
        class_node_index=class_idx,
        num_classes=c,
        num_queries=q,
    )


def build_node_classification_prompt_subgraphs(
    base_g,
    class_text_emb: torch.Tensor,
    noi_text_emb: torch.Tensor,
    query_indices: torch.Tensor,
    prompt_edge_text_emb: Optional[torch.Tensor] = None,
    relations: PromptRelations = OFA_PROMPT_RELATIONS,
    num_hops: int = 2,
    max_nodes_per_hop: int = 100,
    seed: int = 0,
):
    """Build a disjoint batch of OFA-style k-hop prompted query graphs."""
    device = base_g.x.device
    query_indices = query_indices.to(device=device, dtype=torch.long).view(-1)
    class_text_emb = class_text_emb.to(device=device, dtype=torch.float32)
    if class_text_emb.dim() != 2:
        raise ValueError("class_text_emb must have shape [num_classes, llm_dim]")

    q = int(query_indices.numel())
    c = int(class_text_emb.size(0))
    d = int(base_g.x.size(1))
    n = int(base_g.x.size(0))
    if q == 0:
        raise ValueError("query_indices is empty")
    if class_text_emb.size(1) != d:
        raise ValueError(f"class feature width {class_text_emb.size(1)} != node width {d}")
    noi_text_emb = _as_2d_prompt_feature(noi_text_emb, d, device)

    adj = getattr(base_g, "adj", None)
    if adj is None:
        adj = make_csr_adj(base_g.edge_index, n)

    base_edge_feat = getattr(base_g, "original_edge_feature", None)
    if base_edge_feat is None:
        base_edge_feat = getattr(base_g, "edge_attr", None)
        if base_edge_feat is not None:
            base_edge_feat = base_edge_feat[:1]

    x_parts = []
    edge_parts = []
    edge_type_parts = []
    edge_attr_parts = []
    noi_indices = []
    class_indices = []
    query_node_indices = []
    offset = 0
    query_cpu = query_indices.detach().cpu().tolist()

    for pos, root in enumerate(query_cpu):
        neighbors = _sample_fixed_hop_neighbors(
            adj,
            int(root),
            hop=num_hops,
            max_nodes_per_hop=max_nodes_per_hop,
            seed=seed + pos * 100003,
        )
        nodes_np = np.concatenate([np.array([root], dtype=np.int64), neighbors])
        sub_adj = adj[nodes_np, :][:, nodes_np].tocoo()
        feat_count = int(nodes_np.shape[0])
        nodes = torch.as_tensor(nodes_np, dtype=torch.long, device=device)
        sample_x = torch.cat(
            [
                base_g.x[nodes].to(device=device, dtype=torch.float32),
                noi_text_emb,
                class_text_emb,
            ],
            dim=0,
        )
        x_parts.append(sample_x)

        noi_idx = offset + feat_count
        cls_idx = offset + feat_count + 1 + torch.arange(c, device=device, dtype=torch.long)
        query_idx = offset
        noi_indices.append(torch.tensor(noi_idx, device=device, dtype=torch.long))
        class_indices.append(cls_idx)
        query_node_indices.append(torch.tensor(query_idx, device=device, dtype=torch.long))

        row = torch.as_tensor(sub_adj.row, dtype=torch.long, device=device) + offset
        col = torch.as_tensor(sub_adj.col, dtype=torch.long, device=device) + offset
        sample_edges = [torch.stack([row, col], dim=0)]
        sample_types = [
            torch.full((row.numel(),), relations.ORIGINAL, dtype=torch.long, device=device)
        ]
        sample_attrs = [_repeat_edge_feature(base_edge_feat, int(row.numel()), d, device)]

        query_to_noi = torch.tensor([[query_idx], [noi_idx]], dtype=torch.long, device=device)
        noi_to_query = torch.tensor([[noi_idx], [query_idx]], dtype=torch.long, device=device)
        noi_to_class = torch.stack(
            [
                torch.full((c,), noi_idx, dtype=torch.long, device=device),
                cls_idx,
            ],
            dim=0,
        )
        class_to_noi = torch.stack(
            [
                cls_idx,
                torch.full((c,), noi_idx, dtype=torch.long, device=device),
            ],
            dim=0,
        )
        sample_edges.extend([query_to_noi, noi_to_query, noi_to_class, class_to_noi])
        sample_types.extend(
            [
                torch.full((1,), relations.QUERY_TO_NOI, dtype=torch.long, device=device),
                torch.full((1,), relations.NOI_TO_QUERY, dtype=torch.long, device=device),
                torch.full((c,), relations.NOI_TO_CLASS, dtype=torch.long, device=device),
                torch.full((c,), relations.CLASS_TO_NOI, dtype=torch.long, device=device),
            ]
        )
        prompt_edge_count = 2 + 2 * c
        sample_attrs.append(_repeat_edge_feature(prompt_edge_text_emb, prompt_edge_count, d, device))

        edge_parts.append(torch.cat(sample_edges, dim=1))
        edge_type_parts.append(torch.cat(sample_types, dim=0))
        edge_attr_parts.append(torch.cat(sample_attrs, dim=0))
        offset += feat_count + 1 + c

    return SimpleNamespace(
        x=torch.cat(x_parts, dim=0),
        edge_index=torch.cat(edge_parts, dim=1),
        edge_attr=torch.cat(edge_attr_parts, dim=0),
        edge_type=torch.cat(edge_type_parts, dim=0),
        query_node_index=torch.stack(query_node_indices),
        noi_node_index=torch.stack(noi_indices),
        class_node_index=torch.stack(class_indices, dim=0),
        num_classes=c,
        num_queries=q,
    )


class OFAFullNodeClassifier(nn.Module):
    """Prompt-graph OFA node classifier with class-node link scores."""

    def __init__(
        self,
        llm_name: str = "ST",
        emb_dim: int = 768,
        num_layers: int = 7,
        num_rels: int = 5,
        dropout: float = 0.15,
        jk: str = "none",
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if llm_name not in LLM_DIM_DICT:
            raise ValueError(f"unknown llm_name={llm_name}")
        self.llm_name = llm_name
        self.emb_dim = emb_dim
        self.num_rels = num_rels
        self.jk = jk
        self.llm_proj = nn.Linear(LLM_DIM_DICT[llm_name], emb_dim)
        self.encoder = PyGRGCNEdge(
            num_layers=num_layers,
            num_rels=num_rels,
            inp_dim=emb_dim,
            out_dim=emb_dim,
            drop_ratio=dropout,
            JK=jk,
            batch_norm=True,
        )
        self.att = SingleHeadAtt(emb_dim) if jk == "none" else None
        self.score_mlp = MLP([emb_dim, 2 * emb_dim, emb_dim, 1], dropout=head_dropout)

    def forward(
        self,
        base_g,
        class_text_emb: torch.Tensor,
        noi_text_emb: torch.Tensor,
        query_indices: torch.Tensor,
        prompt_edge_text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if getattr(base_g, "prompt_graph_mode", "full") == "subgraph":
            prompt = build_node_classification_prompt_subgraphs(
                base_g,
                class_text_emb,
                noi_text_emb,
                query_indices,
                prompt_edge_text_emb=prompt_edge_text_emb,
                num_hops=int(getattr(base_g, "subgraph_hops", 2)),
                max_nodes_per_hop=int(getattr(base_g, "max_nodes_per_hop", 100)),
                seed=int(getattr(base_g, "subgraph_seed", 0)),
            )
        else:
            prompt = build_node_classification_prompt(
                base_g,
                class_text_emb,
                noi_text_emb,
                query_indices,
                prompt_edge_text_emb=prompt_edge_text_emb,
            )
        projected = SimpleNamespace(
            x=self.llm_proj(prompt.x),
            edge_index=prompt.edge_index,
            edge_attr=self.llm_proj(prompt.edge_attr),
            edge_type=prompt.edge_type,
        )
        node_emb = self.encoder(projected)
        if isinstance(node_emb, list):
            if self.att is None:
                node_emb = node_emb[-1]
            else:
                layer_emb = torch.stack(node_emb, dim=1)
                node_emb = self.att(layer_emb, projected.x.unsqueeze(1), layer_emb)[0].squeeze(1)
        class_emb = node_emb[prompt.class_node_index.reshape(-1)]
        scores = self.score_mlp(class_emb).view(prompt.num_queries, prompt.num_classes)
        return scores

    @torch.no_grad()
    def predict(
        self,
        base_g,
        class_text_emb: torch.Tensor,
        noi_text_emb: torch.Tensor,
        query_indices: torch.Tensor,
        prompt_edge_text_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(base_g, class_text_emb, noi_text_emb, query_indices, prompt_edge_text_emb).argmax(dim=-1)


def build_model(
    llm_name: str = "ST",
    emb_dim: int = 768,
    num_layers: int = 7,
    num_rels: int = 5,
    dropout: float = 0.15,
    jk: str = "none",
    pretrained_encoder: Optional[str] = None,
) -> OFAFullNodeClassifier:
    model = OFAFullNodeClassifier(
        llm_name=llm_name,
        emb_dim=emb_dim,
        num_layers=num_layers,
        num_rels=num_rels,
        dropout=dropout,
        jk=jk,
    )
    if pretrained_encoder:
        if not osp.exists(pretrained_encoder):
            raise FileNotFoundError(f"pretrained_encoder not found: {pretrained_encoder}")
        payload = torch.load(pretrained_encoder, map_location="cpu", weights_only=False)
        enc_info = model.encoder.load_state_dict(payload["encoder"], strict=False)
        proj_info = model.llm_proj.load_state_dict(payload["llm_proj"], strict=False)
        print(
            "[ofa_bridge] loaded encoder/llm_proj from "
            f"{pretrained_encoder} encoder={enc_info} llm_proj={proj_info}",
            flush=True,
        )
    return model


def prepare_base_graph(data, device: Optional[torch.device] = None, *, to_undirected: bool = True):
    """Convert a TAG registry PyG Data object into the raw OFA graph view."""
    if getattr(data, "node_text_feat", None) is None:
        raise ValueError("TAG data is missing node_text_feat")
    x = data.node_text_feat.float()
    edge_index = data.edge_index.long()
    etf = getattr(data, "edge_text_feat", None)
    if etf is None:
        edge_attr = torch.zeros(edge_index.size(1), x.size(1), dtype=x.dtype)
    else:
        etf = etf.float()
        if etf.dim() == 2 and etf.size(0) == edge_index.size(1):
            edge_attr = etf
        else:
            edge_attr = etf[0:1].expand(edge_index.size(1), -1)
    if to_undirected:
        from torch_geometric.utils import to_undirected as pyg_to_undirected

        edge_index, edge_attr = pyg_to_undirected(
            edge_index,
            edge_attr=edge_attr,
            num_nodes=x.size(0),
            reduce="mean",
        )
    edge_type = torch.full((edge_index.size(1),), OFA_PROMPT_RELATIONS.ORIGINAL, dtype=torch.long)
    g = SimpleNamespace(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        adj=make_csr_adj(edge_index, x.size(0)),
        original_edge_feature=edge_attr[:1].contiguous(),
    )
    class_text_emb = data.class_node_text_feat.float()
    if device is not None:
        g.x = g.x.to(device)
        g.edge_index = g.edge_index.to(device)
        g.edge_attr = g.edge_attr.to(device)
        g.edge_type = g.edge_type.to(device)
        class_text_emb = class_text_emb.to(device)
    return g, class_text_emb
