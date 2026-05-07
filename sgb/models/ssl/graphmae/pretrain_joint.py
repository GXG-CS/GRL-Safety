"""GraphMAE joint multi-source pretraining on GFT's 9-dataset mixture.

Mirrors GFT's `config/pt_data.yaml` mixture + NeighborLoader-style mini-batch
sampling, but implemented in DGL via `dgl.batch` + `dgl.dataloading.NeighborSampler`.

Key differences from `pretrain_transductive.py`:
- Loads 9 datasets via `sgb.data.tag_registry.load_tag` (not a single dataset)
- Merges via `dgl.batch([...])` into one disjoint-union graph
- Weighted node sampling via the same formula GFT uses
  (`get_train_node_idx` in `reference/gft_ref/GFT/dataset/process_datasets.py`)
- Mini-batch training with `NeighborSampler([10, 10])` + batch_size=1024
- Forces `decoder_type='mlp'` so the reconstruction decoder is node-wise
  (a GAT decoder on blocks would require maintaining a separate decoder graph)
- Bypasses `GAT.forward` (which assumes a single full graph) and iterates
  `encoder.gat_layers` directly, passing `blocks[l]` per layer

Does not modify any existing GraphMAE files. Features stay on CPU; only the
per-batch sampled features are transferred to GPU.
"""

import os
import sys
import argparse

import numpy as np
import torch
import dgl
import yaml
from torch.optim import AdamW

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sgb.data.tag_registry import load as load_tag  # type: ignore
from graphmae.models.edcoder import PreModel  # type: ignore
from graphmae.utils import set_random_seed  # type: ignore


# -----------------------------------------------------------------------------
# Data: load 9 datasets via tag_registry, merge into one DGL disjoint union
# -----------------------------------------------------------------------------
# IMPORTANT: molecule datasets contain tens of millions of atom-nodes in total
# (chempcba ~11M, chemblpre ~8.8M, chemhiv ~1M). Materializing per-node
# features via `node_text_feat[x]` for all of them gives ~66 GB which is not
# feasible. We instead mirror GFT's approach: keep per-dataset `node_text_feat`
# lookup tables concatenated into ONE global lookup (~260k × 768 ≈ 800 MB), and
# store OFFSETTED long indices into this lookup on each DGL node. Per-batch
# features are materialized inside the training loop only for the sampled
# subgraph.
def load_multi_datasets_dgl(setting):
    """Read `pt_data.yaml` mixture and return a merged DGL graph + global
    feature lookup.

    For every dataset:
      - Chem (chemhiv / chempcba / chemblpre): `data.x` is already a 1D long
        tensor of atom-type indices into `data.node_text_feat` (a small lookup).
      - NC / KG: we treat each node as its own "type" and use
        `arange(num_nodes)` as the index into `data.node_text_feat` (which is
        already [N, 768]).

    All per-dataset indices are then offset by the cumulative size of the
    previous lookup tables so the merged index is globally unique, and all
    `node_text_feat` tables are concatenated into one `global_lookup`.

    Returns:
        big_graph:     dgl.DGLGraph (disjoint union; ndata['x'] holds the
                       offsetted long index into global_lookup)
        global_lookup: [total_types, 768] float32 — one row per unique
                       (dataset, node-type)
        weights:       list[float] — per-dataset sampling weights from yaml
        ptr:           1D LongTensor of cumulative *graph-node* counts
                       [0, n1, n1+n2, ...] — used by the weighted sampler
        names:         list[str]   — dataset names in ptr-segment order
    """
    yaml_path = os.path.abspath(
        os.path.join(_HERE, "..", "gft", "config", "pt_data.yaml")
    )
    with open(yaml_path, "r") as f:
        WEIGHT = yaml.safe_load(f)
    assert setting in WEIGHT, f"setting '{setting}' not in pt_data.yaml keys: {list(WEIGHT.keys())}"
    mixture = WEIGHT[setting]
    dataset_names = list(mixture.keys())
    weights = [float(v) for v in mixture.values()]

    print(f"[joint pretrain] Loading {len(dataset_names)} datasets: {dataset_names}")
    print(f"[joint pretrain] Per-dataset weights: {weights}")

    graphs = []
    lookup_parts = []
    x_offset = 0

    for name in dataset_names:
        data, _ = load_tag(name)

        # Per-dataset feature lookup table (small for chem, large for NC/KG)
        node_text_feat = data.node_text_feat.float()  # [M, 768]

        # Per-dataset x indices into this lookup table
        if data.x is not None and data.x.dtype == torch.long and data.x.ndim == 1:
            # Chem case: x is already 1D long atom-type indices
            x_local = data.x.clone().long()
        else:
            # NC/KG case: each node gets its own unique "type" (arange)
            x_local = torch.arange(node_text_feat.size(0), dtype=torch.long)

        # Offset x by cumulative lookup size so indices are globally unique
        x_global = x_local + x_offset
        num_nodes = x_global.size(0)

        src = data.edge_index[0].long()
        dst = data.edge_index[1].long()
        g = dgl.graph((src, dst), num_nodes=num_nodes)
        g = g.remove_self_loop().add_self_loop()
        g.ndata["x"] = x_global
        graphs.append(g)

        lookup_parts.append(node_text_feat)
        x_offset += node_text_feat.size(0)

        print(
            f"  {name:12s}: {num_nodes:>10d} nodes, {g.num_edges():>12d} edges, "
            f"lookup=+{node_text_feat.size(0)} (running total {x_offset})"
        )

    big_graph = dgl.batch(graphs)
    if "x" not in big_graph.ndata:
        big_graph.ndata["x"] = torch.cat([g.ndata["x"] for g in graphs], dim=0)

    global_lookup = torch.cat(lookup_parts, dim=0).contiguous()

    ptr = torch.tensor([0] + [g.num_nodes() for g in graphs]).cumsum(0)
    total_n = int(ptr[-1].item())
    total_lookup = global_lookup.size(0)
    lookup_mb = global_lookup.numel() * 4 / 1024 / 1024
    print(
        f"[joint pretrain] Merged DGL graph: {total_n} nodes, "
        f"{big_graph.num_edges()} edges"
    )
    print(
        f"[joint pretrain] Global feature lookup: {total_lookup} rows × 768d "
        f"({lookup_mb:.1f} MB)"
    )
    return big_graph, global_lookup, weights, ptr, dataset_names


def get_weighted_train_nids(ptr, weights):
    """Expand node indices per dataset by its sampling weight.

    Mirrors GFT's `get_train_node_idx` logic (`process_datasets.py:186-198`):
      - integer part of weight repeats each node that many times
      - fractional part samples a random fraction of nodes once
    """
    total = []
    for i, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        arr = torch.arange(int(s), int(e))
        w = weights[i]
        int_w = int(w)
        mod_w = w - int_w
        left = arr.repeat(int_w)
        right_count = int(mod_w * arr.size(0))
        right = arr[torch.randperm(arr.size(0))[:right_count]]
        total.append(torch.cat([left, right]))
    return torch.cat(total)


# -----------------------------------------------------------------------------
# Model forward on blocks (bypass GAT.forward which assumes a single graph)
# -----------------------------------------------------------------------------
def encode_blocks(gat_encoder, blocks, feat):
    """Run GraphMAE's GAT encoder on a list of DGL blocks.

    `gat_encoder` is a `GAT` instance whose `.gat_layers` is an nn.ModuleList
    of `GATConv`. `GATConv.forward` already supports blocks (sees
    `graph.is_block` and slices dst features accordingly — see
    `graphmae/models/gat.py:226-229`). We just need to pass one block per
    layer, which `GAT.forward` doesn't do.
    """
    h = feat
    for l, layer in enumerate(gat_encoder.gat_layers):
        h = layer(blocks[l], h)
    return gat_encoder.head(h)


# -----------------------------------------------------------------------------
# Pretrain loop
# -----------------------------------------------------------------------------
def pretrain_joint(args):
    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[joint pretrain] Using {device}")
    set_random_seed(args.seed)

    # ---- Data ----
    big_graph, global_lookup, weights, ptr, names = load_multi_datasets_dgl(args.pretrain_dataset)
    # Graph + index indices stay on CPU; the global feature lookup table is
    # moved to GPU once. Per-batch, we do `global_lookup[input_idx]` on GPU
    # to materialize the sampled subgraph features.
    global_lookup = global_lookup.to(device)

    train_nids = get_weighted_train_nids(ptr, weights)
    print(f"[joint pretrain] weighted train_nids size: {train_nids.size(0)}")

    # ---- Model ----
    # Force MLP decoder so the decoder is node-wise (no decoder graph needed).
    model = PreModel(
        in_dim=args.num_features,
        num_hidden=args.num_hidden,
        num_layers=args.num_layers,
        nhead=args.num_heads,
        nhead_out=args.num_heads,
        activation=args.activation,
        feat_drop=args.in_drop,
        attn_drop=args.attn_drop,
        negative_slope=0.2,
        residual=False,
        norm=None,
        mask_rate=args.mask_rate,
        encoder_type="gat",
        decoder_type="mlp",
        loss_fn=args.loss_fn,
        drop_edge_rate=0.0,
        replace_rate=args.replace_rate,
        alpha_l=args.alpha_l,
        concat_hidden=False,
    ).to(device)

    print(
        f"[joint pretrain] PreModel: encoder=gat decoder=mlp "
        f"hidden={args.num_hidden} layers={args.num_layers} heads={args.num_heads}"
    )

    # ---- Sampler + DataLoader ----
    # CPU-side sampling; features stay on CPU until the batch is ready.
    sampler = dgl.dataloading.NeighborSampler([10] * args.num_layers)
    dataloader = dgl.dataloading.DataLoader(
        big_graph,
        train_nids,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler:
        sched_fn = lambda ep: (1 + np.cos(ep * np.pi / args.max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_fn)
    else:
        scheduler = None

    # ---- Train ----
    for epoch in range(args.max_epoch):
        model.train()
        losses = []
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # Move blocks to GPU; materialize per-batch features via lookup
            blocks = [b.to(device) for b in blocks]
            input_idx = big_graph.ndata["x"][input_nodes].to(device)
            input_feat = global_lookup[input_idx]

            num_output = output_nodes.size(0)
            # DGL convention: first `num_output` rows of `input_feat` correspond
            # to the output (center) nodes.
            target = input_feat[:num_output].clone()

            num_mask = max(1, int(num_output * args.mask_rate))
            perm = torch.randperm(num_output, device=device)
            mask_idx = perm[:num_mask]

            # Replace masked positions with the learnable mask token
            masked_feat = input_feat.clone()
            masked_feat[mask_idx] = model.enc_mask_token.to(device)

            # Encoder on blocks
            enc_rep = encode_blocks(model.encoder, blocks, masked_feat)
            # enc_rep shape: [num_output, num_hidden]

            # Linear projection + MLP decoder (both node-wise)
            rep = model.encoder_to_decoder(enc_rep)
            recon = model.decoder(rep)
            # recon shape: [num_output, in_dim]

            # Reconstruction loss on masked positions only
            loss = model.criterion(recon[mask_idx], target[mask_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 200 == 0:
                print(
                    f"  epoch {epoch+1}/{args.max_epoch} step {step}: "
                    f"loss={loss.item():.4f}  "
                    f"input_nodes={input_nodes.size(0)}  output_nodes={num_output}"
                )

        if scheduler is not None:
            scheduler.step()

        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"[epoch {epoch+1}/{args.max_epoch}] avg_loss={avg:.4f}")

        # ---- Periodic checkpoint (every --save_every epochs) ----
        ckpt_dir = args.ckpt_dir or f"ckpts/graphmae/{args.pretrain_dataset}"
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            interim = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), interim)
            print(f"[joint pretrain] (interim) saved {interim}")

    # ---- Final checkpoint ----
    ckpt_dir = args.ckpt_dir or f"ckpts/graphmae/{args.pretrain_dataset}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[joint pretrain] Final model saved to {ckpt_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("GraphMAE joint multi-source pretrain")
    parser.add_argument("--pretrain_dataset", type=str, default="all",
                        help="Key in sgb/models/gfm/gft/config/pt_data.yaml "
                             "(all / node / link / graph / citation / ...)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Encoder / decoder
    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--num_hidden", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--in_drop", type=float, default=0.2)
    parser.add_argument("--attn_drop", type=float, default=0.1)

    # GraphMAE-specific
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--loss_fn", type=str, default="sce", choices=["sce", "mse"])
    parser.add_argument("--alpha_l", type=float, default=3.0)

    # Optim
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1024)

    # Output
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save intermediate checkpoint every N epochs (0 = disable)")
    args = parser.parse_args()

    pretrain_joint(args)


if __name__ == "__main__":
    main()
