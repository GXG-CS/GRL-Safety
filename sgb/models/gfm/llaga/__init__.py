"""LLaGA wrapper for GFM-Safety benchmark.

Reference: Chen et al., "LLaGA: Large Language and Graph Assistant",
ICML 2024. arxiv.org/abs/2402.08170. Upstream code is cloned at
`reference/llaga_ref/`; this wrapper reimplements the ND-template graph
tokenization + mm_projector idea on top of modern transformers (5.x) so
it plays nicely with the rest of the safety pipeline.
"""

from .model import (
    LlagaConfig,
    LlagaNDEncoder,
    LlagaNDClassifier,
    build_nd_subgraph_indices,
    DEFAULT_GRAPH_PAD_ID,
)

__all__ = [
    "LlagaConfig",
    "LlagaNDEncoder",
    "LlagaNDClassifier",
    "build_nd_subgraph_indices",
    "DEFAULT_GRAPH_PAD_ID",
]
