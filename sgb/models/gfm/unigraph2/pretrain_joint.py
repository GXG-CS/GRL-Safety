"""UniGraph2 joint multi-source pretraining on GFT's 9-dataset mixture.

Design parallel to `sgb/models/ssl/graphmae/pretrain_joint.py`:
- Load 9 datasets via `sgb.data.tag_registry.load`
- Merge into one DGL disjoint-union graph via `dgl.batch`
- Chem datasets keep a per-dataset `node_text_feat` lookup; NC/KG materialize
  768d SBERT directly. All lookups are concatenated into one global table,
  globally-unique long indices stored on `big_graph.ndata['x']`
- Weighted per-dataset node expansion (copies GFT's `pt_data.yaml` formula)

Key difference vs GraphMAE:
- UniGraph2 expects a PROPER DGL subgraph (not bipartite blocks) so that
  `dgl.nn.GATConv` can run with a single graph, and so we can compute SPD
  per-subgraph. We use `NeighborSampler([fanout, ...])` to pick input nodes,
  then call `dgl.node_subgraph(big_graph, input_nodes)` to materialize the
  induced DGL graph at each step.
- Per-subgraph k-hop SPD is computed on CPU via BFS (same as smufang's
  GFMBenchmark/UniGraph2/pretrain.py:compute_spd_matrix), then uploaded once.
- The model signature is `model(graph, {"text": feat}, spd_matrix)` with
  `input_dims={"text": 768}`; single-modality dict because our benchmark
  is pure TAG.

Checkpoint path: `ckpts/unigraph2/<pretrain_dataset>/model.pt`.
"""

import os
import sys
import argparse

import numpy as np
import torch
import dgl
import yaml
from torch.optim import AdamW
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sgb.data.tag_registry import load as load_tag  # type: ignore
from models.unigraph2 import UniGraph2  # type: ignore


# -----------------------------------------------------------------------------
# Data: load GFT-9 datasets, merge into one DGL disjoint union with global
# feature lookup. Copied from graphmae/pretrain_joint.py for consistency.
# -----------------------------------------------------------------------------
def load_multi_datasets_dgl(setting):
    """Read `pt_data.yaml` mixture and return a merged DGL graph + global
    feature lookup.

    Mirrors graphmae/pretrain_joint.py::load_multi_datasets_dgl so the two
    methods see exactly the same pretraining corpus with the same per-dataset
    sampling weights.
    """
    yaml_path = os.path.abspath(
        os.path.join(_HERE, "..", "gft", "config", "pt_data.yaml")
    )
    with open(yaml_path, "r") as f:
        WEIGHT = yaml.safe_load(f)
    assert setting in WEIGHT, (
        f"setting '{setting}' not in pt_data.yaml keys: {list(WEIGHT.keys())}"
    )
    mixture = WEIGHT[setting]
    dataset_names = list(mixture.keys())
    weights = [float(v) for v in mixture.values()]

    print(f"[UG2 joint pretrain] Loading {len(dataset_names)} datasets: {dataset_names}")
    print(f"[UG2 joint pretrain] Per-dataset weights: {weights}")

    graphs = []
    lookup_parts = []
    x_offset = 0

    for name in dataset_names:
        data, _ = load_tag(name)
        node_text_feat = data.node_text_feat.float()  # [M, 768]

        if data.x is not None and data.x.dtype == torch.long and data.x.ndim == 1:
            # Chem: data.x is 1D atom-type index into node_text_feat
            x_local = data.x.clone().long()
        else:
            # NC/KG: each node is its own type
            x_local = torch.arange(node_text_feat.size(0), dtype=torch.long)

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
        f"[UG2 joint pretrain] Merged DGL graph: {total_n} nodes, "
        f"{big_graph.num_edges()} edges"
    )
    print(
        f"[UG2 joint pretrain] Global feature lookup: {total_lookup} rows × 768d "
        f"({lookup_mb:.1f} MB)"
    )
    return big_graph, global_lookup, weights, ptr, dataset_names


def get_weighted_train_nids(ptr, weights):
    """Expand node indices per dataset by its sampling weight.

    Same formula as graphmae/pretrain_joint.py::get_weighted_train_nids,
    which mirrors GFT's `get_train_node_idx`.
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


def get_per_dataset_nids(ptr, k):
    """Sample `k` random node indices from each dataset segment in the big
    graph. Returns a single concatenated 1D LongTensor of size k * num_datasets.

    Used when we want uniform per-dataset representation regardless of size —
    each dataset contributes exactly `k` seeds per epoch. This differs from
    `get_weighted_train_nids` (which inflates by pt_data.yaml weights) and
    makes the total seed count predictable: 9 * k for the GFT-9 mixture.

    Small-data safety: if a segment has fewer than k nodes, we sample with
    replacement to still return k seeds (preserves the flat-per-dataset shape).
    """
    total = []
    for i, (s, e) in enumerate(zip(ptr[:-1], ptr[1:])):
        seg_size = int(e) - int(s)
        if seg_size >= k:
            perm = torch.randperm(seg_size)[:k]
        else:
            # with replacement
            perm = torch.randint(0, seg_size, (k,))
        total.append(perm + int(s))
    return torch.cat(total)


# -----------------------------------------------------------------------------
# SPD computation on a sub-DGLGraph. Clips distances > k to (k+1) and returns
# a [N, N] float tensor on the requested device.
#
# Originally a pure-Python BFS copied from GFMBenchmark/UniGraph2/pretrain.py.
# That version was O(N*E) in Python and became the single biggest bottleneck
# on the 9-dataset pretraining job (dominated wall-clock ~10x over GATConv).
# Replaced with scipy's `shortest_path` (C-level Dijkstra with `unweighted`
# falls back to BFS), which brings per-subgraph SPD from ~1 sec to ~0.05 sec
# at ~1000 nodes. The output is identical up to the k+1 clamp.
# -----------------------------------------------------------------------------
def compute_spd_matrix(graph, k=2, device="cpu"):
    num_nodes = graph.num_nodes()
    src, dst = graph.edges()
    src_np = src.cpu().numpy().astype(np.int32)
    dst_np = dst.cpu().numpy().astype(np.int32)
    # Symmetrize for undirected BFS
    row = np.concatenate([src_np, dst_np])
    col = np.concatenate([dst_np, src_np])
    data = np.ones(row.shape[0], dtype=np.float32)
    adj = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

    dist = shortest_path(adj, directed=False, unweighted=True)
    # np.inf for unreachable pairs → clip to k+1; also clip finite > k to k+1
    dist[np.isinf(dist)] = k + 1
    dist[dist > k] = k + 1
    return torch.from_numpy(dist).float().to(device)


# -----------------------------------------------------------------------------
# Checkpoint save/load helpers
#
# Full checkpoints carry optimizer + scheduler + epoch + RNG state so a
# crashed or preempted run can resume exactly where it left off. Downstream
# FT scripts (finetune_*.py) already handle both formats — they check for
# `isinstance(state, dict) and "model" in state` and unwrap accordingly —
# so switching pretraining to full checkpoints is backwards compatible.
# -----------------------------------------------------------------------------
def save_checkpoint(path, epoch, model, optimizer, scheduler, args):
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_torch": torch.get_rng_state(),
        "rng_np": np.random.get_state(),
        "args": vars(args),
    }
    if torch.cuda.is_available():
        ckpt["rng_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Back-compat: if someone passes a bare state_dict, load model only and
    # return epoch=0 (fresh restart with just the weights).
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        model.load_state_dict(ckpt)
        print(f"[UG2 joint pretrain] Loaded bare state_dict from {path}; "
              f"resuming from epoch 0 (no optimizer/scheduler state).")
        return 0
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("rng_torch") is not None:
        torch.set_rng_state(ckpt["rng_torch"])
    if ckpt.get("rng_np") is not None:
        np.random.set_state(ckpt["rng_np"])
    if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception as e:
            print(f"[UG2 joint pretrain] WARN could not restore CUDA RNG: {e}")
    start_epoch = int(ckpt.get("epoch", 0))
    print(
        f"[UG2 joint pretrain] Resumed from {path}, starting at epoch "
        f"{start_epoch + 1}/{ckpt.get('args', {}).get('max_epoch', '?')}"
    )
    return start_epoch


# -----------------------------------------------------------------------------
# Pretrain loop
# -----------------------------------------------------------------------------
def pretrain_joint(args):
    device = (
        torch.device(f"cuda:{args.gpu}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[UG2 joint pretrain] Using {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data ----
    big_graph, global_lookup, weights, ptr, names = load_multi_datasets_dgl(
        args.pretrain_dataset
    )
    global_lookup = global_lookup.to(device)

    # ---- Model ----
    # Single text modality, 768d SBERT input matches hidden_dim.
    model = UniGraph2(
        input_dims={"text": args.num_features},
        hidden_dim=args.num_hidden,
        num_experts=args.num_experts,
        num_selected_experts=args.num_selected_experts,
        num_layers=args.num_layers,
        feat_drop_rate=args.feat_drop_rate,
        edge_mask_rate=args.edge_mask_rate,
        gamma=args.gamma,
        lambda_spd=args.lambda_spd,
    ).to(device)

    print(
        f"[UG2 joint pretrain] UniGraph2: hidden={args.num_hidden} "
        f"layers={args.num_layers} experts={args.num_experts}/"
        f"{args.num_selected_experts} gamma={args.gamma} "
        f"lambda_spd={args.lambda_spd}"
    )

    sampler = dgl.dataloading.NeighborSampler([args.fanout] * args.num_layers)

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.scheduler:
        sched_fn = lambda ep: (1 + np.cos(ep * np.pi / args.max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_fn)
    else:
        scheduler = None

    # ---- Resume (optional) ----
    start_epoch = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume path does not exist: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        if start_epoch >= args.max_epoch:
            print(
                f"[UG2 joint pretrain] Checkpoint epoch {start_epoch} >= "
                f"max_epoch {args.max_epoch}; nothing to resume. Exiting."
            )
            return

    # ---- Train ----
    # Sampling strategy:
    #   - `seeds_per_dataset` (preferred for UniGraph2): each epoch draws a
    #     fresh random `k` seeds from each dataset segment. 9 datasets ×
    #     `seeds_per_dataset` = total seeds per epoch. Chem datasets do NOT
    #     dominate because every dataset contributes the same count. This
    #     is what we use in practice because per-step cost (SPD + GATConv on
    #     a materialized subgraph) is 100× GraphMAE's, so a much smaller
    #     total seed budget is needed to fit in a 12h slot.
    #   - `nids_per_epoch` (legacy fallback): keeps GFT's pt_data.yaml
    #     weighting then subsamples to this cap. Only used when
    #     `seeds_per_dataset` is unset.
    #   - neither set: full weighted pool (~4.4M on GFT-9, infeasible for
    #     UniGraph2 in 12h — will not finish).
    #
    # The DataLoader is rebuilt every epoch so each epoch's seed draw is
    # fresh (via the sampler constructor, not shuffle over a fixed pool).
    for epoch in range(start_epoch, args.max_epoch):
        if args.seeds_per_dataset is not None and args.seeds_per_dataset > 0:
            train_nids = get_per_dataset_nids(ptr, args.seeds_per_dataset)
        else:
            train_nids = get_weighted_train_nids(ptr, weights)
            if args.nids_per_epoch is not None and args.nids_per_epoch > 0 \
                    and args.nids_per_epoch < train_nids.size(0):
                perm = torch.randperm(train_nids.size(0))[: args.nids_per_epoch]
                train_nids = train_nids[perm]
        if epoch == 0:
            if args.seeds_per_dataset is not None and args.seeds_per_dataset > 0:
                print(
                    f"[UG2 joint pretrain] seeds_per_dataset={args.seeds_per_dataset} "
                    f"→ {train_nids.size(0)} total seeds/epoch "
                    f"({len(names)} datasets × {args.seeds_per_dataset})"
                )
            else:
                print(
                    f"[UG2 joint pretrain] train_nids per epoch: {train_nids.size(0)}"
                )
        dataloader = dgl.dataloading.DataLoader(
            big_graph,
            train_nids,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )

        model.train()
        losses = []
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # Induced subgraph on the sampled node set. Relabeled to 0..|V|-1.
            # dgl.NID stores original big_graph ids so we can look up features.
            subg = dgl.node_subgraph(big_graph, input_nodes)
            subg = subg.remove_self_loop().add_self_loop()

            if subg.num_nodes() > args.max_subgraph_nodes:
                # Ego-subgraph from `input_nodes[0]` (center seed) — prevents
                # rare worst-case SPD blow-up on densely connected NC graphs.
                continue

            # Materialize subgraph features via global lookup
            input_idx = big_graph.ndata["x"][input_nodes].to(device)
            input_feat = global_lookup[input_idx]

            # SPD on CPU (BFS), then used inside model via _compute_spd_loss
            # which samples pairs and moves them to GPU.
            spd = compute_spd_matrix(subg, k=args.spd_k, device="cpu")

            subg = subg.to(device)

            loss, _ = model(subg, {"text": input_feat}, spd)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if step % 50 == 0:
                print(
                    f"  epoch {epoch+1}/{args.max_epoch} step {step}: "
                    f"loss={loss.item():.4f}  "
                    f"input_nodes={input_nodes.size(0)}  subg_nodes={subg.num_nodes()}"
                )

        if scheduler is not None:
            scheduler.step()

        avg = float(np.mean(losses)) if losses else float("nan")
        print(f"[UG2 epoch {epoch+1}/{args.max_epoch}] avg_loss={avg:.4f}")

        ckpt_dir = args.ckpt_dir or f"ckpts/unigraph2/{args.pretrain_dataset}"
        os.makedirs(ckpt_dir, exist_ok=True)
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0):
            interim = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pt")
            save_checkpoint(interim, epoch + 1, model, optimizer, scheduler, args)
            print(f"[UG2 joint pretrain] (interim) saved {interim}")

    # ---- Final checkpoint ----
    ckpt_dir = args.ckpt_dir or f"ckpts/unigraph2/{args.pretrain_dataset}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    save_checkpoint(ckpt_path, args.max_epoch, model, optimizer, scheduler, args)
    print(f"[UG2 joint pretrain] Final model saved to {ckpt_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("UniGraph2 joint multi-source pretrain")
    parser.add_argument(
        "--pretrain_dataset", type=str, default="all",
        help="Key in sgb/models/gfm/gft/config/pt_data.yaml",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # Encoder
    parser.add_argument("--num_features", type=int, default=768)
    parser.add_argument("--num_hidden", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--num_selected_experts", type=int, default=2)

    # UniGraph2-specific
    parser.add_argument("--feat_drop_rate", type=float, default=0.1)
    parser.add_argument("--edge_mask_rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--lambda_spd", type=float, default=0.5)
    parser.add_argument("--spd_k", type=int, default=2)

    # Sampler
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument(
        "--max_subgraph_nodes", type=int, default=2000,
        help="Skip steps whose induced subgraph has more nodes than this. "
             "Keeps SPD computation tractable on dense NC graphs (arxiv, products).",
    )

    # Optim
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--max_epoch", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--nids_per_epoch", type=int, default=None,
        help="Legacy fallback: cap weighted-pool seeds/epoch. Ignored when "
             "--seeds_per_dataset is set.",
    )
    parser.add_argument(
        "--seeds_per_dataset", type=int, default=None,
        help="Preferred sampler: draw exactly K seeds from each dataset per "
             "epoch, giving (num_datasets * K) seeds total. Makes per-epoch "
             "cost predictable and prevents chem-atom-level datasets from "
             "dominating. Recommended: 1500 for UniGraph2 (9*1500=13500 seeds "
             "= ~420 steps/epoch at batch=32).",
    )

    # Output
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint (.pt) to resume from. Restores model + "
             "optimizer + scheduler + RNG + epoch counter, then continues "
             "training until --max_epoch. Accepts either the full-dict "
             "format written by save_checkpoint() or a bare model "
             "state_dict (legacy — restarts from epoch 0 with weights only).",
    )
    args = parser.parse_args()

    pretrain_joint(args)


if __name__ == "__main__":
    main()
