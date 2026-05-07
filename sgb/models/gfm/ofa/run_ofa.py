#!/usr/bin/env python
"""Run isolated OFA-full prompt-graph node-classification experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

_PROJECT_ROOT_BOOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", ".."))
if _PROJECT_ROOT_BOOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_BOOT)

from sgb.data.tag_registry import load as load_tag
from sgb.models.gfm.ofa.ofa_train_utils import (
    accuracy,
    build_degree_ood_split,
    compute_node_degree,
    graph_view,
    indices_to_mask,
    mask_to_idx,
    normalize_masks,
    per_class_recall,
    prompt_logits,
    step_imbalance_mask,
    train_prompt_model,
)
from sgb.metrics.fairness import compute_group_fairness


RUN_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
FN_SEVERITIES = [(0, 0.0), (1, 0.1), (2, 0.25), (3, 0.5), (4, 1.0), (5, 2.0)]
ED_SEVERITIES = [(0, 0.0), (1, 0.05), (2, 0.10), (3, 0.20), (4, 0.30), (5, 0.50)]
RHOS = [5, 10, 20]


def _append_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        new_file = f.tell() == 0
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        for row in rows:
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _load_data(args):
    if args.tag_cache_root:
        return load_tag(args.dataset, cache_root=args.tag_cache_root)[0]
    return load_tag(args.dataset)[0]


def _feature_noise(x: torch.Tensor, train_mask: torch.Tensor, sigma_rel: float, seed: int) -> torch.Tensor:
    if sigma_rel == 0.0:
        return x
    tm = train_mask.bool().to(x.device)
    std = x[tm].std(dim=0, keepdim=True)
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + sigma_rel * std * eps


def _drop_edges(g, p: float, seed: int):
    if p <= 0.0:
        return g.edge_index, g.edge_attr, g.edge_type
    gen = torch.Generator(device=g.edge_index.device).manual_seed(int(seed))
    keep = torch.rand(g.edge_index.size(1), generator=gen, device=g.edge_index.device) > p
    return g.edge_index[:, keep], g.edge_attr[keep], g.edge_type[keep]


def run_clean(args):
    data = _load_data(args)
    rows = []
    for si, (tm, vm, testm) in enumerate(normalize_masks(data, args.n_splits)):
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        test_acc = accuracy(
            model, g, cls_emb, noi_emb, pedge_emb, labels, testm.to(args.device),
            batch_size=args.eval_query_batch_size,
        )
        rows.append({
            "dataset": args.dataset,
            "method": "ofa",
            "split_idx": si,
            "seed": RUN_SEEDS[si],
            "val_acc": best_val,
            "test_acc": test_acc,
        })
        print(
            f"[OFA-FULL-CLEAN] {args.dataset} split={si} val={best_val:.4f} "
            f"test={test_acc:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def run_fn(args):
    data = _load_data(args)
    rows = []
    for si, (tm, vm, testm) in enumerate(normalize_masks(data, args.n_splits)):
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        for sev_id, sigma in FN_SEVERITIES:
            gv = graph_view(g, x=_feature_noise(g.x, tm, sigma, 1000 * RUN_SEEDS[si] + sev_id))
            test_acc = accuracy(
                model, gv, cls_emb, noi_emb, pedge_emb, labels, testm.to(args.device),
                batch_size=args.eval_query_batch_size,
            )
            rows.append({
                "dataset": args.dataset,
                "method": "ofa",
                "split_idx": si,
                "seed": RUN_SEEDS[si],
                "severity_id": sev_id,
                "sigma_rel": sigma,
                "val_acc": best_val,
                "test_acc": test_acc,
            })
        print(
            f"[OFA-FULL-FN] {args.dataset} split={si} clean={rows[-6]['test_acc']:.4f} "
            f"sev5={rows[-1]['test_acc']:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def run_ed(args):
    data = _load_data(args)
    rows = []
    for si, (tm, vm, testm) in enumerate(normalize_masks(data, args.n_splits)):
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        for sev_id, drop in ED_SEVERITIES:
            ei, ea, et = _drop_edges(g, drop, 1000 * RUN_SEEDS[si] + sev_id)
            gv = graph_view(g, edge_index=ei, edge_attr=ea, edge_type=et)
            test_acc = accuracy(
                model, gv, cls_emb, noi_emb, pedge_emb, labels, testm.to(args.device),
                batch_size=args.eval_query_batch_size,
            )
            rows.append({
                "dataset": args.dataset,
                "method": "ofa",
                "split_idx": si,
                "seed": RUN_SEEDS[si],
                "severity_id": sev_id,
                "drop_rate": drop,
                "val_acc": best_val,
                "test_acc": test_acc,
            })
        print(
            f"[OFA-FULL-ED] {args.dataset} split={si} clean={rows[-6]['test_acc']:.4f} "
            f"sev5={rows[-1]['test_acc']:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def run_imb(args):
    data = _load_data(args)
    labels_cpu = data.y.squeeze().long()
    num_classes = int(labels_cpu.max().item()) + 1
    tm0, vm0, testm0 = normalize_masks(data, 1)[0]
    rows, pc_rows = [], []
    for rho in RHOS:
        for si in range(args.n_seeds):
            imb_train, minor_classes = step_imbalance_mask(tm0, labels_cpu, rho, split_seed=si)
            t0 = time.time()
            model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
                data, imb_train, vm0, seed=RUN_SEEDS[si], device=args.device, args=args, val_metric="bacc"
            )
            test_idx = mask_to_idx(testm0, args.device)
            logits = prompt_logits(
                model, g, cls_emb, noi_emb, pedge_emb, test_idx,
                batch_size=args.eval_query_batch_size,
            )
            rec = per_class_recall(logits, labels, test_idx, num_classes)
            major_mask = np.array([(c not in minor_classes) for c in range(num_classes)])
            major_rec = float(np.nanmean(rec[major_mask])) if major_mask.any() else float("nan")
            minor_rec = float(np.nanmean(rec[~major_mask])) if (~major_mask).any() else float("nan")
            bacc = float(np.nanmean(rec))
            rows.append({
                "dataset": args.dataset,
                "method": "ofa",
                "rho": rho,
                "seed": si,
                "val_bacc": best_val,
                "bacc": bacc,
                "major_recall": major_rec,
                "minor_recall": minor_rec,
                "num_minor_classes": len(minor_classes),
            })
            for c in range(num_classes):
                pc_rows.append({
                    "dataset": args.dataset,
                    "method": "ofa",
                    "rho": rho,
                    "seed": si,
                    "class": c,
                    "recall": None if np.isnan(rec[c]) else float(rec[c]),
                    "is_minor": c in minor_classes,
                })
            print(
                f"[OFA-FULL-IMB] {args.dataset} rho={rho} seed={si} bacc={bacc:.4f} "
                f"major={major_rec:.4f} minor={minor_rec:.4f} t={time.time() - t0:.1f}s",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)
    if args.per_class_csv:
        _append_csv(args.per_class_csv, pc_rows)


def run_struct(args):
    data = _load_data(args)
    labels_cpu = data.y.squeeze().long()
    deg = compute_node_degree(data.edge_index, labels_cpu.numel()).to(args.device).float()
    head_mask = deg > deg.median()
    tail_mask = deg <= deg.median()
    rows = []
    for si, (tm, vm, testm) in enumerate(normalize_masks(data, args.n_seeds)):
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        test_dev = testm.to(args.device).bool()
        acc_all = accuracy(model, g, cls_emb, noi_emb, pedge_emb, labels, test_dev, batch_size=args.eval_query_batch_size)
        acc_head = accuracy(model, g, cls_emb, noi_emb, pedge_emb, labels, test_dev & head_mask, batch_size=args.eval_query_batch_size)
        acc_tail = accuracy(model, g, cls_emb, noi_emb, pedge_emb, labels, test_dev & tail_mask, batch_size=args.eval_query_batch_size)
        rows.append({
            "dataset": args.dataset,
            "method": "ofa",
            "split_seed": si,
            "val_acc": best_val,
            "test_acc": acc_all,
            "acc_head": acc_head,
            "acc_tail": acc_tail,
            "gap_head_minus_tail": acc_head - acc_tail,
            "n_head": int((test_dev & head_mask).sum().item()),
            "n_tail": int((test_dev & tail_mask).sum().item()),
        })
        print(
            f"[OFA-FULL-STRUCT] {args.dataset} split={si} all={acc_all:.4f} "
            f"head={acc_head:.4f} tail={acc_tail:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def run_fair(args):
    from sgb.metrics.fairness import load_tolokers_education_binary

    data = _load_data(args)
    sens, _meta = load_tolokers_education_binary()
    rows = []
    for si, (tm, vm, testm) in enumerate(normalize_masks(data, args.n_seeds)):
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        all_idx = torch.arange(labels.numel(), device=args.device)
        logits = prompt_logits(model, g, cls_emb, noi_emb, pedge_emb, all_idx, batch_size=args.eval_query_batch_size)
        prob = torch.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1).detach().cpu().numpy()
        prob_np = prob.detach().cpu().numpy()
        y_np = labels.detach().cpu().numpy()
        test_np = testm.bool().numpy()
        sens_np = sens.numpy().astype(int)
        score = prob_np[:, 1] if prob_np.shape[1] > 1 else prob_np.squeeze()
        metrics = compute_group_fairness(y_np[test_np], pred[test_np], score[test_np], sens_np[test_np])
        rows.append({"dataset": args.dataset, "method": "ofa", "seed": si, "val_acc": best_val, **metrics})
        print(
            f"[OFA-FULL-FAIR] {args.dataset} seed={si} dSP={metrics['delta_sp']:.4f} "
            f"dEO={metrics['delta_eo']:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)



def run_ood_degree(args):
    data = _load_data(args)
    labels_cpu = data.y.squeeze().long()
    n_nodes = labels_cpu.numel()
    rows = []
    for si in range(args.n_seeds):
        split = build_degree_ood_split(edge_index=data.edge_index, labels=labels_cpu, split_seed=si)
        if split["meta"]["degree_shift"] == "not_applicable":
            continue
        tm = indices_to_mask(split["train"], n_nodes)
        vm = indices_to_mask(split["id_val"], n_nodes)
        idm = indices_to_mask(split["id_test"], n_nodes)
        oodm = indices_to_mask(split["ood_test"], n_nodes)
        t0 = time.time()
        model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
            data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
        )
        id_acc = accuracy(model, g, cls_emb, noi_emb, pedge_emb, labels, idm.to(args.device), batch_size=args.eval_query_batch_size)
        ood_acc = accuracy(model, g, cls_emb, noi_emb, pedge_emb, labels, oodm.to(args.device), batch_size=args.eval_query_batch_size)
        rows.append({
            "dataset": args.dataset,
            "method": "ofa",
            "split_seed": si,
            "id_val": best_val,
            "id_test_acc": id_acc,
            "ood_test_acc": ood_acc,
            "gap": id_acc - ood_acc,
            "degree_protocol": split["meta"].get("strategy", "good_60_20_20_descending"),
        })
        print(
            f"[OFA-FULL-OOD] {args.dataset} split={si} id={id_acc:.4f} "
            f"ood={ood_acc:.4f} t={time.time() - t0:.1f}s",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def _time_naive(year: torch.Tensor):
    y = year.squeeze().long()
    n = int(y.numel())
    order = torch.argsort(y * n + torch.arange(n))
    n_train = int(round(n * 0.60))
    n_val = int(round(n * 0.05))
    return {
        "train": order[:n_train],
        "id_val": order[n_train:n_train + n_val],
        "ood_test": order[n_train + n_val:],
        "meta": {"protocol": "naive", "strategy": "good_60_5_35_ascending_year"},
    }


def _time_aggressive(year: torch.Tensor, split_seed: int):
    y = year.squeeze().long()
    train_idx = (y <= 2010).nonzero(as_tuple=True)[0]
    test_idx = (y >= 2017).nonzero(as_tuple=True)[0]
    rng = np.random.RandomState(split_seed)
    perm = torch.as_tensor(rng.permutation(int(train_idx.numel())), dtype=torch.long)
    cut = int(0.9 * train_idx.numel())
    return {
        "train": train_idx[perm[:cut]],
        "id_val": train_idx[perm[cut:]],
        "ood_test": test_idx,
        "meta": {"protocol": "aggressive", "strategy": "year_cutoff_train<=2010_ood>=2017"},
    }


def run_ood_time(args):
    data = _load_data(args)
    if not hasattr(data, "node_year") or data.node_year is None:
        raise RuntimeError(f"{args.dataset} has no node_year; temporal OOD is not available")
    labels_cpu = data.y.squeeze().long()
    n_nodes = labels_cpu.numel()
    rows = []
    builders = [
        ("naive", lambda seed: _time_naive(data.node_year)),
        ("aggressive", lambda seed: _time_aggressive(data.node_year, seed)),
    ]
    for protocol, builder in builders:
        for si in range(args.n_seeds):
            split = builder(si)
            tm = indices_to_mask(split["train"], n_nodes)
            vm = indices_to_mask(split["id_val"], n_nodes)
            oodm = indices_to_mask(split["ood_test"], n_nodes)
            t0 = time.time()
            model, best_val, g, cls_emb, noi_emb, pedge_emb, labels = train_prompt_model(
                data, tm, vm, seed=RUN_SEEDS[si], device=args.device, args=args
            )
            id_acc = accuracy(
                model, g, cls_emb, noi_emb, pedge_emb, labels, vm.to(args.device),
                batch_size=args.eval_query_batch_size,
            )
            ood_acc = accuracy(
                model, g, cls_emb, noi_emb, pedge_emb, labels, oodm.to(args.device),
                batch_size=args.eval_query_batch_size,
            )
            rows.append({
                "dataset": args.dataset,
                "method": "ofa",
                "protocol": protocol,
                "split_seed": si,
                "id_val": best_val,
                "id_test_acc": id_acc,
                "ood_test_acc": ood_acc,
                "gap": id_acc - ood_acc,
                "strategy": split["meta"]["strategy"],
            })
            print(
                f"[OFA-FULL-OOD-TIME] {args.dataset} protocol={protocol} split={si} "
                f"id={id_acc:.4f} ood={ood_acc:.4f} gap={id_acc - ood_acc:+.4f} "
                f"t={time.time() - t0:.1f}s",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    _append_csv(args.output_csv, rows)


def _default_output(project_root: str, axis: str) -> str:
    rel = {
        "clean": "experiments/ofa/clean/ofa_clean.csv",
        "fn": "experiments/ofa/corruption/fn/ofa_fn.csv",
        "ed": "experiments/ofa/corruption/ed/ofa_ed.csv",
        "imb": "experiments/ofa/imbalance/ofa_imb.csv",
        "struct": "experiments/ofa/fairness/structural/ofa_struct.csv",
        "fair": "experiments/ofa/fairness/demographic/ofa_fair.csv",
        "ood_degree": "experiments/ofa/ood/degree/ofa_ood_degree.csv",
        "ood_time": "experiments/ofa/ood/time/ofa_ood_time.csv",
    }[axis]
    return osp.join(project_root, rel)


def _resolve(path: str | None, project_root: str) -> str | None:
    if not path:
        return None
    if osp.isabs(path):
        return path
    return osp.join(project_root, path)


def _existing_default(*paths: str) -> str | None:
    for p in paths:
        if p and osp.exists(p):
            return p
    return None


def main():
    project_root = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "..", ".."))
    default_cache = osp.join(project_root, "datasets", "TAG")
    default_model_cache = osp.join(project_root, "cache_data", "model")
    default_ckpt = _existing_default(
        osp.join(project_root, "ckpts", "ofa", "pretrain_e2e", "encoder_weights_final.pt"),
        osp.join(project_root, "sgb", "models", "gfm", "ofa", "ckpts", "pretrain_e2e", "encoder_weights_final.pt"),
    )

    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=["clean", "fn", "ed", "imb", "struct", "fair", "ood_degree", "ood_time"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--emb_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=7)
    p.add_argument("--num_rels", type=int, default=5)
    p.add_argument("--jk", default="none")
    p.add_argument("--llm_name", default="ST")
    p.add_argument("--train_query_batch_size", type=int, default=256)
    p.add_argument("--eval_query_batch_size", type=int, default=256)
    p.add_argument("--max_train_query", type=int, default=0)
    p.add_argument("--prompt_graph_mode", default="subgraph", choices=["subgraph", "full"])
    p.add_argument("--subgraph_hops", type=int, default=2)
    p.add_argument("--max_nodes_per_hop", type=int, default=100)
    p.add_argument("--freeze_encoder", action="store_true")
    p.add_argument("--top_k_pct", type=float, default=0.10)
    p.add_argument("--max_test_nodes", type=int, default=100)
    p.add_argument("--tag_cache_root", default=default_cache)
    p.add_argument("--model_cache_dir", default=default_model_cache)
    p.add_argument("--pretrained_encoder", default=default_ckpt)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--per_class_csv", default=None)
    args = p.parse_args()

    args.project_root = project_root
    args.device = _device(args.gpu)
    args.pretrained_encoder = _resolve(args.pretrained_encoder, project_root)
    if args.output_csv is None:
        args.output_csv = _default_output(project_root, args.axis)
    if args.per_class_csv is None and args.axis == "imb":
        args.per_class_csv = osp.join(project_root, "experiments/ofa/imbalance/ofa_imb_per_class.csv")

    print(
        f"[OFA-FULL] axis={args.axis} dataset={args.dataset} device={args.device} "
        f"ckpt={args.pretrained_encoder} tag_cache={args.tag_cache_root}",
        flush=True,
    )

    if args.axis == "clean":
        run_clean(args)
    elif args.axis == "fn":
        run_fn(args)
    elif args.axis == "ed":
        run_ed(args)
    elif args.axis == "imb":
        run_imb(args)
    elif args.axis == "struct":
        run_struct(args)
    elif args.axis == "fair":
        run_fair(args)
    elif args.axis == "ood_degree":
        run_ood_degree(args)
    elif args.axis == "ood_time":
        run_ood_time(args)


if __name__ == "__main__":
    main()
