"""GFT scaffold-OOD eval on BBBP / BACE.

Wraps GFT's existing graph-FT pipeline (`run_feature_noise.run`) by:
  1. Monkey-patching `_load_graph_dataset` to return scaffold or random split
  2. Disabling the noise loop (empty SEVERITIES)
  3. Re-tagging clean test AUC lines as `[SCAFFOLD_RAW]`
  4. Running twice (random + scaffold) and computing per-method gap

This keeps GFT's full TaskModel + proto + codebook FT machinery intact;
only the data split is swapped.
"""
import argparse
import os.path as osp
import sys
import re
import io
import contextlib

import numpy as np
import torch

_GFT_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_GFT_DIR, "..", "..", "..", ".."))
if _GFT_DIR not in sys.path:
    sys.path.insert(0, _GFT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import run_feature_noise as gft_fn  # noqa: E402
from sgb.data.scaffold_split import load_splits  # noqa: E402

_FN_RAW_RE = re.compile(r"\[FN_RAW\][^\n]*?seed=(?P<seed>\d+)\s+sev=0\s+\S+\s+test_acc=(?P<auc>[-\d.]+)")


def _patched_loader_factory(orig_loader, dataset_name, split_type):
    """Return a function with same signature that overrides the split."""
    def patched(name, root):
        dataset, _orig_split, labels, num_classes, num_tasks = orig_loader(name, root)
        n = len(dataset)
        sp = load_splits(dataset_name, n)[split_type]
        new_split = {
            "train": sp["train"].cpu().numpy(),
            "valid": sp["val"].cpu().numpy(),
            "test": sp["test"].cpu().numpy(),
        }
        print(f"[GFT-Scaffold-Patch] split_type={split_type} "
              f"train={len(new_split['train'])} val={len(new_split['valid'])} "
              f"test={len(new_split['test'])}")
        return dataset, new_split, labels, num_classes, num_tasks
    return patched


def _run_one(params, dataset_name, split_type):
    """Run GFT FT once with the given split_type; return list of test AUCs."""
    orig_loader = gft_fn._load_graph_dataset
    orig_severities = gft_fn.SEVERITIES
    gft_fn._load_graph_dataset = _patched_loader_factory(orig_loader, dataset_name, split_type)
    gft_fn.SEVERITIES = []  # skip noise loop

    # GFT's run() calls wandb.finish() at the end; re-init for each call.
    import wandb
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project=f"gft-scaffold-{split_type}", mode="offline", reinit=True)

    class _Tee:
        def __init__(self, stream, buf):
            self.stream, self.buf = stream, buf
        def write(self, s):
            self.stream.write(s)
            self.stream.flush()
            self.buf.write(s)
        def flush(self):
            self.stream.flush()

    buf = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = _Tee(real_stdout, buf)
    try:
        gft_fn.run(params)
    finally:
        sys.stdout = real_stdout
        gft_fn._load_graph_dataset = orig_loader
        gft_fn.SEVERITIES = orig_severities

    out = buf.getvalue()

    aucs = []
    for m in _FN_RAW_RE.finditer(out):
        aucs.append(float(m.group("auc")))
    return aucs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["bbbp", "bace", "tox21"])
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--config_yaml", default=None,
                   help="Override config; defaults to GFT's standard.yaml")
    args, unknown = p.parse_known_args()

    # Build params via GFT's own arg parser, but force what we need.
    sys.argv = ["run_scaffold_ood.py",
                "--use_params",
                "--dataset", args.dataset,
                "--finetune_dataset", args.dataset]
    if args.ckpt_dir:
        sys.argv += ["--ckpt_dir", args.ckpt_dir]
    sys.argv += unknown

    from utils.args import get_args_finetune
    params = get_args_finetune()

    # Load yaml config (GFT's `main` reads from sgb/models/gfm/gft/config/finetune.yaml)
    import yaml
    config_path = args.config_yaml or osp.abspath(
        osp.join(_GFT_DIR, "..", "config", "finetune.yaml")
    )
    with open(config_path) as f:
        defaults = yaml.safe_load(f)
    if "base" in defaults:
        params.update(defaults["base"])
    # Per-dataset overrides under graph: bbbp: {...}
    g_section = defaults.get("graph", {})
    if isinstance(g_section, dict) and args.dataset in g_section:
        params.update(g_section[args.dataset])

    params["task"] = "graph"
    params["finetune_dataset"] = args.dataset
    params["dataset"] = args.dataset
    params["repeat"] = args.n_seeds
    params["seeds"] = list(range(args.n_seeds))
    # Mirror chemhiv's hyperparams for bbbp/bace (smaller, similar mol task)
    chemhiv_defaults = g_section.get("chemhiv", {}) if isinstance(g_section, dict) else {}
    for k in ("normalize", "finetune_epochs", "early_stop", "batch_size",
              "finetune_lr", "lambda_proto", "lambda_act", "trade_off",
              "num_instances_per_class", "n_train"):
        if k in chemhiv_defaults and k not in g_section.get(args.dataset, {}):
            params[k] = chemhiv_defaults[k]
    # Override with shorter epochs for scaffold-OOD (quick FT)
    params.setdefault("finetune_epochs", 100)
    params.setdefault("early_stop", 20)
    if args.ckpt_dir:
        params["ckpt_dir"] = args.ckpt_dir

    # Suppress wandb (use offline)
    import wandb
    if not getattr(wandb, "_initialized", False):
        wandb.init(project="gft-scaffold", mode="offline")

    print(f"[GFT FT-Scaffold] device=cuda dataset={args.dataset} n_seeds={args.n_seeds}")

    rand_aucs = _run_one(params, args.dataset, "random")
    print(f"[GFT FT-Scaffold] random AUCs: {rand_aucs}")

    scaf_aucs = _run_one(params, args.dataset, "scaffold")
    print(f"[GFT FT-Scaffold] scaffold AUCs: {scaf_aucs}")

    n = min(len(rand_aucs), len(scaf_aucs))
    if n == 0:
        print("[ERROR] no AUC parsed from GFT output")
        return
    for i in range(n):
        print(f"[SCAFFOLD_RAW] method=GFT dataset={args.dataset} "
              f"split_type=random seed={i} test_auc={rand_aucs[i]:.4f} test_f1=0.0")
        print(f"[SCAFFOLD_RAW] method=GFT dataset={args.dataset} "
              f"split_type=scaffold seed={i} test_auc={scaf_aucs[i]:.4f} test_f1=0.0")
    rand_arr = np.asarray(rand_aucs[:n])
    scaf_arr = np.asarray(scaf_aucs[:n])
    gap = float(rand_arr.mean() - scaf_arr.mean())
    print(f"[SCAFFOLD_AGG] method=GFT dataset={args.dataset} "
          f"random_auc_mean={rand_arr.mean():.4f} random_auc_std={rand_arr.std():.4f} "
          f"scaffold_auc_mean={scaf_arr.mean():.4f} scaffold_auc_std={scaf_arr.std():.4f} "
          f"gap={gap:.4f} n_seeds={n}")
    print("[METRIC] auc_roc")


if __name__ == "__main__":
    main()
