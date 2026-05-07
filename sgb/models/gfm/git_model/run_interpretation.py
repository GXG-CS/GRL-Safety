"""V2: subgraph-ablation interpretation for GIT (PyG-style)."""
import argparse
import copy
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

_HERE = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_HERE, "..", "..", ".."))
for p in (_HERE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from sgb.models.gfm.git_model.run_interpretation import (
    FTModel, _get_splits, train_ft, load_encoder_ckpt)
from model.encoder import Encoder
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.interpretation_node import compute_node_fidelity

RUN_SEEDS = [42, 43, 44]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cora")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--hidden_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--backbone", default="sage")
    p.add_argument("--activation", default="relu")
    p.add_argument("--normalize", default="none")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--max_test_nodes", type=int, default=100)
    p.add_argument("--K_hop", type=int, default=2)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--explainers", default="grad,random")
    p.add_argument("--topk_list", default="0.05,0.10,0.20,0.50")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GIT V2-INTERP] device={device}, dataset={args.dataset}")

    data, _ = load_tag(args.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat
    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    N = int(data.num_nodes)
    input_dim = data.x.size(1)
    x_clean = data.x.detach().clone()
    base_ei = data.edge_index.detach().clone()
    print(f"[GIT V2-INTERP] N={N} d={input_dim} C={num_classes} E={base_ei.size(1)}")

    splits = _get_splits(data)
    n_seeds = 1 if args.debug else args.n_seeds
    max_test = 5 if args.debug else args.max_test_nodes
    explainers = [s.strip() for s in args.explainers.split(",") if s.strip()]
    topk_list = [float(s) for s in args.topk_list.split(",") if s.strip()]

    agg = {(e, tk): {"fp": [], "fm": [], "ch": [], "n_eval": []}
           for e in explainers for tk in topk_list}

    tm, vm, tsm = splits[0]
    tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)

    for rs in RUN_SEEDS[:n_seeds]:
        torch.manual_seed(rs); np.random.seed(rs)
        activation_cls = nn.ReLU if args.activation == "relu" else nn.LeakyReLU
        encoder = Encoder(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            activation=activation_cls, num_layers=args.num_layers,
            backbone=args.backbone, normalize=args.normalize, dropout=args.dropout,
        )
        encoder, _res = load_encoder_ckpt(encoder, args.ckpt_dir)
        model = FTModel(encoder, args.hidden_dim, num_classes, dropout=args.dropout).to(device)
        model = train_ft(model, x_clean, base_ei, y, tm, vm,
                         args.lr, args.wd, args.max_epochs, args.patience)
        model.eval()
        with torch.no_grad():
            pred = model(x_clean, base_ei).argmax(-1)
        acc = accuracy_score(y[tsm].cpu().numpy(), pred[tsm].cpu().numpy()) * 100.0
        print(f"[GIT V2-INTERP] seed={rs} clean_acc={acc:.2f}")

        def forward_fn(edge_index_mod, x_arg, _model=model):
            return _model(x_arg, edge_index_mod)

        test_idx = tsm.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        rng = np.random.RandomState(rs)
        sample = rng.choice(test_idx, size=min(max_test, len(test_idx)), replace=False)
        sample = torch.tensor(sorted(sample.tolist()))

        for explainer in explainers:
            out = compute_node_fidelity(
                model=model, x=x_clean, edge_index=base_ei, y=y,
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
                print(f"[INTERP_NODE_V2_RAW] method=GIT dataset={args.dataset} "
                      f"split=0 seed={rs} explainer={explainer} topk={tk} "
                      f"fid_plus={r['fid_plus_mean']:.4f} fid_minus={r['fid_minus_mean']:.4f} "
                      f"char={r['char_mean']:.4f} n_eval={r['n_eval']}")

    for (explainer, tk), d in agg.items():
        if not d["fp"]:
            continue
        fp = np.array(d["fp"]); fm = np.array(d["fm"]); ch = np.array(d["ch"])
        print(f"[INTERP_NODE_V2_AGG] method=GIT dataset={args.dataset} "
              f"explainer={explainer} topk={tk} n_runs={len(d['fp'])} "
              f"fid_plus=\"{fp.mean():.4f} ± {fp.std(ddof=1) if len(fp)>1 else 0:.4f}\" "
              f"fid_minus=\"{fm.mean():.4f} ± {fm.std(ddof=1) if len(fm)>1 else 0:.4f}\" "
              f"char=\"{ch.mean():.4f} ± {ch.std(ddof=1) if len(ch)>1 else 0:.4f}\" "
              f"n_eval_mean={int(np.mean(d['n_eval']))}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
