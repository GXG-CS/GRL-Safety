# GRL-Safety

Benchmark code for **"On the Safety of Graph Representation Learning"**.

GRL-Safety evaluates twelve graph representation learning methods across five
safety axes under standardized inputs, splits, stress operators, and metrics,
while preserving each method's native adaptation pipeline.

## Quick start

```bash
conda env create -f environment.yaml
conda activate grl-safety
pip install -e .
```

All Python entry points assume the project root is on `PYTHONPATH`. Either run
them as modules from the repo root, or set `PYTHONPATH=$PWD` before running a
script directly.

## Repo layout

```
GRL-Safety/
├── environment.yaml
├── README.md
└── sgb/
    ├── data/                  TAG dataset registry, per-axis splits and runners
    │   ├── tag_registry.py        load(name) for all 25 TAG datasets
    │   ├── registry.py            task type and metadata per dataset
    │   ├── ood_splits.py          degree shift split
    │   ├── scaffold_split.py      Bemis-Murcko scaffold split for molecules
    │   ├── inductive_kg_split.py  inductive entity split for KG link prediction
    │   ├── imbalance_splits.py    step-imbalance long-tail subsampling
    │   ├── scaffold_ood_runner.py shared GC scaffold-OOD evaluation loop
    │   ├── inductive_kg_runner.py shared LP inductive-KG evaluation loop
    │   └── graph_interpretation_runner.py shared graph-level fidelity loop
    │
    ├── metrics/               shared metric and loader helpers
    │   ├── fairness.py            demographic and structural fairness metrics
    │   ├── interpretation_node.py GraphFramEx node-level fidelity
    │   ├── interpretation_graph.py GraphFramEx graph-level fidelity
    │   └── mutag_motif.py         MUTAG motif ground-truth masks
    │
    └── models/                12 methods organized by tier
        ├── topology/shallow_baseline/      DeepWalk, Node2vec
        ├── supervised/{gcn,gat,sage}.py    GCN, GAT, GraphSAGE
        ├── supervised/gnn_baseline/        shared per-axis runners for the supervised tier
        ├── ssl/bgrl/                       BGRL
        ├── ssl/graphmae/                   GraphMAE
        └── gfm/{gft,git_model,unigraph2,ofa,llaga}/
                                            GFT, GIT, UniGraph2, OFA, LLaGA
```

## Methods (12 across 4 tiers)

| Tier | Method | Reference |
|---|---|---|
| Topology only | DeepWalk | Perozzi et al., 2014 |
| | Node2vec | Grover & Leskovec, 2016 |
| Supervised GNN | GCN | Kipf & Welling, 2017 |
| | GAT | Velickovic et al., 2018 |
| | GraphSAGE | Hamilton et al., 2017 |
| Self-supervised | BGRL | Thakoor et al., 2022 |
| | GraphMAE | Hou et al., 2022 |
| Graph foundation | GFT | Wang et al., 2024 |
| | GIT | Wang et al., 2024 |
| | UniGraph2 | He et al., 2025 |
| | OFA | Liu et al., 2024 |
| | LLaGA (Vicuna-7B) | Chen et al., 2024 |

## Datasets (25 text-attributed graphs)

All datasets are sourced from TSGFM (Chen et al., 2024) and TAGLAS
(Feng et al., 2024). Every node, edge, atom, or relation text field is
embedded with Sentence-BERT into a shared 768-dimensional feature space.

| Domain | Task | Datasets |
|---|---|---|
| Academic | NC | Cora, CiteSeer, PubMed, arXiv, arXiv23, arXivYear, DBLP |
| E-commerce | NC | Elec-Computers, ElePhoto, SportsFit, AmazonRatings |
| Book | NC | BookChild, BookHis |
| Web platform | NC | WikiCS, Tolokers |
| Knowledge graph | LP | WN18RR, FB15K237 |
| Recommendation | LP | ML1M |
| Molecule | GC | BACE, BBBP, ChemHIV, CYP450, MUV, Tox21, ToxCast |

Two additional molecular corpora (ChemBLPre, ChemPCBA) are used at pretraining
time only and are not counted among the 25 evaluation datasets.

## Safety axes (5)

| Axis | Sub-conditions | Tasks | Primary metric |
|---|---|---|---|
| Corruption | feature noise (5 severities), structure noise (5 drop rates) | NC, LP, GC | accuracy / AUC drop |
| OOD | degree shift, temporal shift, scaffold shift, inductive entity shift | NC, GC, LP | ID-OOD drop, MRR / Hits@10 |
| Imbalance | step-imbalance at rho in {5, 10, 20} | NC | balanced accuracy, macro-F1, per-class recall |
| Fairness | structural (head vs tail by degree), demographic (Tolokers education) | NC | head-tail accuracy gap, ΔSP, ΔEO, ΔUtility |
| Interpretation | edge-saliency subgraph ablation, atom-saliency node ablation | NC, GC | GraphFramEx Fid+, Fid-, char score lift |

Per-method, per-axis evaluation coverage follows Table A.1 of the paper.
Topology-only methods are evaluated on structure-noise corruption and on the
degree and temporal OOD shifts only. Interpretation is evaluated on the eight
feature-consuming non-LLM methods.

## Pretraining (9-dataset corpus)

The two self-supervised methods and the five graph foundation models share a
nine-graph pretraining corpus (Cora, PubMed, arXiv, WikiCS, WN18RR, FB15K237,
ChemHIV, ChemBLPre, ChemPCBA), following the joint mixture of GFT.
Each method optimizes its native pretraining objective on this shared corpus
with hyperparameters from Appendix A.1 of the paper.

| Method | Pretrain entry |
|---|---|
| BGRL | `sgb/models/ssl/bgrl/pretrain_joint.py` |
| GraphMAE | `sgb/models/ssl/graphmae/pretrain_joint.py` |
| GFT | `sgb/models/gfm/gft/GFT/pretrain.py` (config in `gfm/gft/config/`) |
| GIT | `sgb/models/gfm/git_model/pretrain.py` (config in `gfm/git_model/config/`) |
| UniGraph2 | `sgb/models/gfm/unigraph2/pretrain_joint.py` |
| OFA | `sgb/models/gfm/ofa/pretrain.py` (config `OFA/pretrain_gft9_config.yaml`) |
| LLaGA | `sgb/models/gfm/llaga/pretrain_projector.py` (Vicuna-7B frozen, only the projector is trained) |

The supervised baselines (GCN, GAT, GraphSAGE) and topology-only methods
(DeepWalk, Node2vec) have no pretraining stage and are trained directly on
each downstream graph.

## Running an evaluation

Each method exposes one entry script per applicable safety sub-condition:

```
sgb/models/<tier>/<method>/run_<axis>.py
```

For example, GFT on feature-noise corruption:

```bash
python -m sgb.models.gfm.gft.GFT.run_feature_noise --dataset cora
```

BGRL on degree-shift OOD:

```bash
python -m sgb.models.ssl.bgrl.run_ood_degree --dataset arxiv
```

OFA dispatches all node-level safety axes through a single CLI
(`sgb/models/gfm/ofa/run_ofa.py`) with `--axis {clean,fn,ed,ood_degree,
ood_time,imb,struct,fair}`. Scaffold-shift and inductive-KG evaluation use the
dedicated scripts in the same directory.

Common flags across runners include `--dataset`, `--seed`, and an output CSV
path. All results are reported as mean ± std across n=5 random seeds.

## Citation

If you use GRL-Safety in your work, please cite:

```bibtex
@misc{grlsafety2026,
  title  = {On the Safety of Graph Representation Learning},
  year   = {2026},
  url    = {https://github.com/GXG-CS/GRL-Safety}
}
```

## License

Code in this repository is released under its respective upstream licenses.
Each `models/<tier>/<method>/` subdirectory keeps the original project's
LICENSE and README files when present, in accordance with the upstream open
source terms (MIT, Apache 2.0, or BSD).

## Acknowledgements

This codebase builds on the public implementations of DeepWalk, Node2vec,
GCN, GAT, GraphSAGE, BGRL, GraphMAE, GFT, GIT, UniGraph2, OFA, and LLaGA,
and adopts dataset packaging from TSGFM and TAGLAS. We thank all upstream
authors for releasing their code and data.
