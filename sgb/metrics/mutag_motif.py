"""MUTAG ground-truth motif masks for plausibility evaluation.

MUTAG node attribute encoding (TUDataset):
    7-dim one-hot atom type, indices = [C, N, O, F, I, Cl, Br]
    (per Debnath 1991 + standard PyG TUDataset README)

MUTAG edge attribute encoding:
    4-dim one-hot bond type, indices = [aromatic, single, double, triple]

Mutagenicity-causing motif (per Debnath 1991, Ying GNNExplainer 2019):
    - NO2 (nitro group): N atom double-bonded to two O atoms
    - NH2 (amine group): N atom single-bonded to two H (H not in MUTAG)

Since MUTAG omits hydrogens, NH2 detection requires inferring H from
valence; here we approximate by tagging any N atom whose neighbors are
all C (terminal N), which captures most NH2 cases.

For each graph we return a boolean mask of size [n_atoms] marking
atoms that are part of either:
  (a) a NO2 group: any N + its two double-bonded O atoms
  (b) a terminal N (potential NH2): any N adjacent only to C atoms

The mask is the union across all such substructures in the molecule.

This avoids the rdkit dependency: we work directly with the TUDataset
atom-type one-hot and edge-type one-hot.
"""
from __future__ import annotations

import os.path as osp
import numpy as np
import torch

ATOM_C, ATOM_N, ATOM_O, ATOM_F, ATOM_I, ATOM_CL, ATOM_BR = range(7)
BOND_AROM, BOND_SINGLE, BOND_DOUBLE, BOND_TRIPLE = range(4)


def _atom_idx(x_oh: torch.Tensor) -> int:
    """Convert one-hot row to atom-type index. Returns -1 if degenerate."""
    if x_oh.sum() == 0:
        return -1
    return int(x_oh.argmax().item())


def _bond_idx(e_oh: torch.Tensor) -> int:
    if e_oh.sum() == 0:
        return -1
    return int(e_oh.argmax().item())


def find_motif_atoms(x_oh: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: torch.Tensor | None) -> torch.Tensor:
    """Return bool mask [n_atoms] = True if atom is part of NO2 or terminal-N."""
    n = x_oh.size(0)
    mask = torch.zeros(n, dtype=torch.bool)
    atom_types = [_atom_idx(x_oh[a]) for a in range(n)]

    # adjacency with bond type
    nbrs = {a: [] for a in range(n)}  # a -> list of (nbr, bond_type)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for k, (s, d) in enumerate(zip(src, dst)):
        bt = _bond_idx(edge_attr[k]) if edge_attr is not None else BOND_SINGLE
        nbrs[s].append((d, bt))

    for a in range(n):
        if atom_types[a] != ATOM_N:
            continue
        # case a) NO2: N with at least 2 double-bond O neighbors
        dbl_o = [nb for (nb, bt) in nbrs[a]
                 if bt == BOND_DOUBLE and atom_types[nb] == ATOM_O]
        if len(dbl_o) >= 2:
            mask[a] = True
            for o in dbl_o:
                mask[o] = True
            continue
        # case b) Terminal N (proxy for NH2): N adjacent only to C
        nbr_atoms = [atom_types[nb] for (nb, _) in nbrs[a]]
        if nbr_atoms and all(at == ATOM_C for at in nbr_atoms) and len(nbr_atoms) <= 1:
            mask[a] = True
    return mask


def load_mutag_motif_masks(root: str | None = None) -> dict[int, torch.Tensor]:
    """Build {graph_idx: bool atom mask} for all MUTAG graphs.

    Returns only entries where motif is non-empty (else key absent).
    """
    from torch_geometric.datasets import TUDataset
    if root is None:
        proj_root = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
        root = osp.join(proj_root, "datasets", "tudataset")
    ds = TUDataset(root=root, name="MUTAG")
    out = {}
    for gi, data in enumerate(ds):
        mask = find_motif_atoms(data.x.float(), data.edge_index,
                                data.edge_attr.float() if data.edge_attr is not None else None)
        if mask.any():
            out[gi] = mask
    return out


if __name__ == "__main__":
    masks = load_mutag_motif_masks()
    sizes = [int(m.sum().item()) for m in masks.values()]
    n_atoms = [int(m.numel()) for m in masks.values()]
    fracs = [s / a for s, a in zip(sizes, n_atoms) if a > 0]
    print(f"[mutag_motif] motif found in {len(masks)} graphs")
    print(f"[mutag_motif] avg motif atoms: {np.mean(sizes):.2f} +- {np.std(sizes):.2f}")
    print(f"[mutag_motif] avg fraction: {np.mean(fracs):.3f}")
