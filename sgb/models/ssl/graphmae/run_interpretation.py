"""V2: subgraph-ablation interpretation for GraphMAE (DGL-internal)."""
import argparse
import copy
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from sklearn.metrics import accuracy_score

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.ssl.graphmae.run_interpretation import (
    build_joint_model, FTModel, load_dataset, train_ft)
from sgb.metrics.interpretation_node import compute_node_fidelity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--max_epochs", default=500, type=int)
    p.add_argument("--patience", default=100, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--wd", default=1e-4, type=float)
    p.add_argument("--dropout", default=0.2, type=float)
    p.add_argument("--max_test_nodes", default=100, type=int)
    p.add_argument("--K_hop", default=2, type=int)
    p.add_argument("--n_seeds", default=3, type=int)
    p.add_argument("--explainers", default="grad,random")
    p.add_argument("--topk_list", default="0.05,0.10,0.20,0.50")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g, feat, y, splits = load_dataset(args.dataset, device)
    num_classes = int(y.max().item()) + 1
    N = feat.size(0); input_size = feat.size(1)
    print(f"[GraphMAE V2-INTERP] N={N} d={input_size} C={num_classes}")

    state = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    # Reconstruct base edge_index from DGL graph (no self-loops for ablation)
    src_g, dst_g = g.edges()
    keep_no_sl = src_g != dst_g
    base_ei = torch.stack([src_g[keep_no_sl], dst_g[keep_no_sl]], dim=0).long().to(device)

    n_seeds = 1 if args.debug else args.n_seeds
    max_test = 5 if args.debug else args.max_test_nodes
    explainers = [s.strip() for s in args.explainers.split(",") if s.strip()]
    topk_list = [float(s) for s in args.topk_list.split(",") if s.strip()]

    RUN_SEEDS = [42, 43, 44]
    tm, vm, tsm = splits[0]
    tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
    agg = {(e, tk): {"fp": [], "fm": [], "ch": [], "n_eval": []}
           for e in explainers for tk in topk_list}

    for rs in RUN_SEEDS[:n_seeds]:
        torch.manual_seed(rs); np.random.seed(rs)
        pre_model = build_joint_model(num_features=input_size)
        pre_model.load_state_dict(state)
        model = FTModel(pre_model, 768, num_classes, args.dropout).to(device)
        model = train_ft(model, g, feat, y, tm, vm, device,
                          args.max_epochs, args.patience, args.lr, args.wd)
        model.eval()
        with torch.no_grad():
            pred = model(g, feat).argmax(-1)
        acc = accuracy_score(y[tsm].cpu().numpy(), pred[tsm].cpu().numpy()) * 100.0
        print(f"[GraphMAE V2-INTERP] seed={rs} clean_acc={acc:.2f}")

        def forward_fn(edge_index_mod, x_arg, _model=model):
            s, d = edge_index_mod[0], edge_index_mod[1]
            keep = s != d
            g_mod = dgl.graph((s[keep].cpu(), d[keep].cpu()),
                              num_nodes=N).remove_self_loop().add_self_loop().to(device)
            return _model(g_mod, x_arg)

        test_idx = tsm.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        rng = np.random.RandomState(rs)
        sample = rng.choice(test_idx, size=min(max_test, len(test_idx)), replace=False)
        sample = torch.tensor(sorted(sample.tolist()))

        for explainer in explainers:
            out = compute_node_fidelity(
                model=model, x=feat, edge_index=base_ei, y=y,
                test_idx=sample, device=device, forward_fn=forward_fn,
                explainer=explainer, topk_list=topk_list,
                target="pred", K_hop=args.K_hop, seed=rs)
            for tk in topk_list:
                r = out["per_topk"][tk]
                if r.get("n_eval", 0) == 0:
                    continue
                agg[(explainer, tk)]["fp"].append(r["fid_plus_mean"])
                agg[(explainer, tk)]["fm"].append(r["fid_minus_mean"])
                agg[(explainer, tk)]["ch"].append(r["char_mean"])
                agg[(explainer, tk)]["n_eval"].append(r["n_eval"])
                print(f"[INTERP_NODE_V2_RAW] method=GraphMAE dataset={args.dataset} "
                      f"split=0 seed={rs} explainer={explainer} topk={tk} "
                      f"fid_plus={r['fid_plus_mean']:.4f} fid_minus={r['fid_minus_mean']:.4f} "
                      f"char={r['char_mean']:.4f} n_eval={r['n_eval']}")

    for (explainer, tk), d in agg.items():
        if not d["fp"]:
            continue
        fp = np.array(d["fp"]); fm = np.array(d["fm"]); ch = np.array(d["ch"])
        print(f"[INTERP_NODE_V2_AGG] method=GraphMAE dataset={args.dataset} "
              f"explainer={explainer} topk={tk} n_runs={len(d['fp'])} "
              f"fid_plus=\"{fp.mean():.4f} ± {fp.std(ddof=1) if len(fp)>1 else 0:.4f}\" "
              f"fid_minus=\"{fm.mean():.4f} ± {fm.std(ddof=1) if len(fm)>1 else 0:.4f}\" "
              f"char=\"{ch.mean():.4f} ± {ch.std(ddof=1) if len(ch)>1 else 0:.4f}\" "
              f"n_eval_mean={int(np.mean(d['n_eval']))}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
