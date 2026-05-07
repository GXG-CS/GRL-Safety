#!/usr/bin/env python
"""OFA joint pretraining on the GFT-9 TAG corpus.

Mirrors the joint-pretrain recipe used by the other GFMs in this benchmark
(GFT, GIT, UniGraph2, LLaGA): one shared 9-dataset corpus, native objective
(prompt-graph binary link prediction) optimized end-to-end.

The 9 datasets follow the GFT pt_data mixture:
    cora, pubmed, arxiv, wikics                 (NC)
    WN18RR, FB15K237                            (LP / KG)
    chemhiv, chemblpre, chempcba                (GC)

Hyperparameters follow paper Appendix A.1:
    num_layers=7, num_hidden=768, dropout=0.15,
    lr=1e-3, num_epochs=50, batch_size=128,
    num_relations=5, JK=none, optimizer=AdamW.

Internally dispatches to ``OFA/run_cdm.py`` with
``--override OFA/pretrain_gft9_config.yaml``.
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import subprocess
import sys

_HERE = osp.dirname(osp.abspath(__file__))
_OFA_DIR = osp.join(_HERE, "OFA")
_DEFAULT_CONFIG = osp.join(_OFA_DIR, "pretrain_gft9_config.yaml")


def main() -> int:
    p = argparse.ArgumentParser(
        description="OFA joint pretraining on the GFT-9 TAG corpus.",
    )
    p.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Override yaml; defaults to OFA/pretrain_gft9_config.yaml.",
    )
    p.add_argument(
        "--ckpt_dir",
        default=None,
        help="Output directory for the pretrained encoder weights.",
    )
    args, extra = p.parse_known_args()

    cmd = [sys.executable, "run_cdm.py", "--override", args.config]
    if args.ckpt_dir is not None:
        cmd += ["--ckpt_dir", args.ckpt_dir]
    cmd += list(extra)

    env = os.environ.copy()
    env["PYTHONPATH"] = _OFA_DIR + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, cwd=_OFA_DIR, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
