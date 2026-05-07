"""Shared utilities for safety-axis evaluation.

Provides metric and loader helpers used by per-method finetune scripts under
``sgb/models/``.

Modules:
  * fairness.py                -  demographic (tolokers) + structural fairness
  * interpretation_node.py  -  node-level explanation fidelity
  * interpretation_graph.py -  graph-level explanation fidelity
  * mutag_motif.py             -  MUTAG motif masks for graph interpretation
"""
