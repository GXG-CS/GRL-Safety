"""Resolve class-label strings for any GFT-9-style dataset that LLaGA will
be pretrained on. Our TAG registry only embeds `class_node_text_feat`
(SBERT vectors) for wikics/arxiv; the raw strings live in OFA's
single_graph data. This module returns the list of class names in
index order so pretrain_projector can tokenise them.
"""

from __future__ import annotations

import json
import os.path as osp
import re
from typing import List, Optional


_REF_ROOT = osp.abspath(
    osp.join(osp.dirname(__file__), "..", "..", "..", "..", "reference", "ofa_ref")
)


def _arxiv_labels() -> List[str]:
    """Parse arxiv_CS_categories.txt; each class entry starts with 'cs.XX (Full Name)'."""
    path = osp.join(_REF_ROOT, "data", "single_graph", "arxiv", "arxiv_CS_categories.txt")
    labels: List[str] = []
    with open(path) as f:
        for line in f:
            m = re.match(r"^cs\.[A-Z]+\s*\(([^)]+)\)", line.strip())
            if m:
                labels.append(m.group(1).strip())
    # ogbn-arxiv has 40 classes; the order in OFA file matches the y-index.
    if len(labels) != 40:
        raise RuntimeError(f"arxiv label parse got {len(labels)} entries, expected 40")
    return labels


def _wikics_labels() -> List[str]:
    path = osp.join(_REF_ROOT, "data", "single_graph", "wikics", "metadata.json")
    with open(path) as f:
        meta = json.load(f)
    labels_dict = meta["labels"]
    return [labels_dict[str(i)] for i in range(len(labels_dict))]


def _kg_relation_list(name: str, n_expected: int) -> List[str]:
    """Reconstruct OFA's `rel_list` ordering by replaying their
    read_knowledge_graph loop over {train,valid,test}.txt."""
    base = osp.join(_REF_ROOT, "data", "KG", name)
    rel_list: List[str] = []
    seen = set()
    for split in ("train", "valid", "test"):
        path = osp.join(base, f"{split}.txt")
        with open(path) as f:
            for line in f.read().split("\n")[:-1]:
                parts = line.split()
                if len(parts) != 3:
                    continue
                r = parts[1]
                if r not in seen:
                    seen.add(r)
                    rel_list.append(r)
    if len(rel_list) != n_expected:
        raise RuntimeError(f"{name}: parsed {len(rel_list)} relations, "
                           f"expected {n_expected}")
    return rel_list


def _wn18rr_labels() -> List[str]:
    # 11 relations: _hypernym, _derivationally_related_form, _instance_hypernym, ...
    # Light textual cleanup so they read like natural labels.
    raw = _kg_relation_list("WN18RR", 11)
    return [r.lstrip("_").replace("_", " ") for r in raw]


def _fb15k237_labels() -> List[str]:
    # 237 relations in path form like /location/country/form_of_government.
    # Keep the leaf+parent tokens for brevity.
    raw = _kg_relation_list("FB15K237", 237)
    out = []
    for r in raw:
        # Drop leading slash and dots-joined internal paths -> readable form
        parts = [seg.replace("_", " ").strip() for seg in r.strip("/").split("/")]
        parts = [p for p in parts if p]
        out.append(" / ".join(parts))
    return out


def _tsgfm_labels(name: str) -> List[str]:
    """Parse TSGFM categories.csv (Amazon product subcategory names)."""
    import csv
    path = osp.join(_REF_ROOT, "..", "TSGFM", "data", "single_graph",
                    name, "categories.csv")
    path = osp.abspath(path)
    if not osp.exists(path):
        raise RuntimeError(f"{name}: categories.csv not found at {path}")
    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append(row["name"].strip())
    return out


def _chem_labels(task: str) -> List[str]:
    """Hard-coded binary labels for chem property prediction.

    OFA chem tasks are multi-label binary (1/0 per task). At the single-task
    prompt level we only need two class strings, plus the task name for prompt
    context (set elsewhere).
    """
    # chemhiv: 1 task 'HIV inhibitor?'
    # chempcba: 128 tasks (PubChem BioAssay)
    # chemblpre: 1310 tasks (ChEMBL pretraining assays)
    # All binary -> yes/no labels
    return ["no", "yes"]


def _tolokers_labels() -> List[str]:
    # Binary: 0 = active / good worker, 1 = banned worker
    return ["active worker", "banned worker"]


def _amazonratings_labels() -> List[str]:
    # 5-bucket product ratings (1–5 stars)
    return ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]


def _dblp_labels() -> List[str]:
    # 4 research areas (common DBLP-4 taxonomy)
    return ["Database", "Data Mining", "Artificial Intelligence", "Information Retrieval"]


_HANDLERS = {
    "arxiv": _arxiv_labels,
    "arxiv23": _arxiv_labels,   # same 40 CS subject areas
    "wikics": _wikics_labels,
    "WN18RR": _wn18rr_labels,
    "FB15K237": _fb15k237_labels,
    "tolokers": _tolokers_labels,
    "dblp": _dblp_labels,
    "amazonratings": _amazonratings_labels,
    "elephoto": lambda: _tsgfm_labels("elephoto"),
    "elecomp": lambda: _tsgfm_labels("elecomp"),
    "bookhis": lambda: _tsgfm_labels("bookhis"),
    "bookchild": lambda: _tsgfm_labels("bookchild"),
    "sportsfit": lambda: _tsgfm_labels("sportsfit"),
    "chemhiv": lambda: _chem_labels("chemhiv"),
    "chempcba": lambda: _chem_labels("chempcba"),
    "chemblpre": lambda: _chem_labels("chemblpre"),
}


def get_label_names(dataset: str, data_obj=None) -> Optional[List[str]]:
    """Return class-name list for `dataset`, or None if we can't find one.

    Preference order:
      1. data_obj.label_names attribute (cora, pubmed in our registry)
      2. OFA reference files (arxiv, wikics)
    """
    if data_obj is not None and getattr(data_obj, "label_names", None):
        names = list(data_obj.label_names)
        if names:
            return [str(x) for x in names]
    if dataset in _HANDLERS:
        return _HANDLERS[dataset]()
    return None
