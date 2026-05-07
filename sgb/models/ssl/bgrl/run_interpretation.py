"""V2: subgraph-ablation interpretation for BGRL (PyG-style)."""
import copy
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from sklearn.metrics import accuracy_score

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.interpretation_node import compute_node_fidelity

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset.')
flags.DEFINE_string('ckpt_path', None, 'BGRL encoder ckpt.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 100, 'Patience.')
flags.DEFINE_float('lr', 5e-4, 'LR.')
flags.DEFINE_float('weight_decay', 1e-5, 'WD.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')
flags.DEFINE_integer('max_test_nodes', 100, 'Test nodes for v2.')
flags.DEFINE_integer('K_hop', 2, 'Receptive field hops.')
flags.DEFINE_integer('n_seeds', 3, 'Run seeds.')
flags.DEFINE_string('explainers', 'grad,random', 'Explainers.')
flags.DEFINE_string('topk_list', '0.05,0.10,0.20,0.50', 'Topk fractions.')
flags.DEFINE_bool('debug', False, 'smoke.')

SPLIT_SEEDS = [0]
RUN_SEEDS = [42, 43, 44]


class FTModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        h = self.encoder(data)
        h = self.dropout(h)
        return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool); m[idx] = True; return m


def _get_splits(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        n = min(5, len(data.train_masks))
        return [(data.train_masks[i].bool(), data.val_masks[i].bool(),
                 data.test_masks[i].bool()) for i in range(n)]
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits; N = data.num_nodes
        tm = _idx_to_mask(s['train'], N); vm = _idx_to_mask(s.get('valid', s.get('val')), N); tsm = _idx_to_mask(s['test'], N)
        return [(tm, vm, tsm)]
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        return [(tm[:, 0].bool(), vm[:, 0].bool(),
                 (tsm[:, 0] if tsm.dim() == 2 else tsm).bool())]
    return [(tm.bool(), vm.bool(), tsm.bool())]


def train_ft(model, data, y, tm, vm, device):
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(1, FLAGS.max_epochs + 1):
        model.train(); optim.zero_grad()
        logits = model(data)
        F.cross_entropy(logits[tm], y[tm]).backward()
        optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(-1)
            val_acc = (pred[vm] == y[vm]).float().mean().item() * 100.0
        if val_acc > best_val:
            best_val = val_acc; best_state = copy.deepcopy(model.state_dict()); no_imp = 0
        else:
            no_imp += 1
            if no_imp >= FLAGS.patience: break
    model.load_state_dict(best_state)
    return model


def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BGRL V2-INTERP] device={device} dataset={FLAGS.dataset}")
    data, _ = load_tag(FLAGS.dataset)
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
    input_size = data.x.size(1)
    print(f"[BGRL V2-INTERP] N={N} d={input_size} C={num_classes}")

    ckpt_state = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and "model" in ckpt_state and isinstance(ckpt_state["model"], dict):
        ckpt_state = ckpt_state["model"]

    splits = _get_splits(data)
    n_seeds = 1 if FLAGS.debug else FLAGS.n_seeds
    max_test = 5 if FLAGS.debug else FLAGS.max_test_nodes
    explainers = [s.strip() for s in FLAGS.explainers.split(",") if s.strip()]
    topk_list = [float(s) for s in FLAGS.topk_list.split(",") if s.strip()]

    x_clean = data.x.detach().clone()
    base_ei = data.edge_index.detach().clone()

    agg = {(e, tk): {"fp": [], "fm": [], "ch": [], "n_eval": []}
           for e in explainers for tk in topk_list}

    tm, vm, tsm = splits[0]
    tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
    for rs in RUN_SEEDS[:n_seeds]:
        torch.manual_seed(rs); np.random.seed(rs)
        encoder = GCN([input_size] + list(FLAGS.graph_encoder_layer), batchnorm=True)
        encoder.load_state_dict(ckpt_state)
        encoder.representation_size = FLAGS.graph_encoder_layer[-1]
        model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)
        data.x = x_clean.detach().clone()
        data.edge_index = base_ei.detach().clone()
        model = train_ft(model, data, y, tm, vm, device)
        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(-1)
        acc = accuracy_score(y[tsm].cpu().numpy(), pred[tsm].cpu().numpy()) * 100.0
        print(f"[BGRL V2-INTERP] seed={rs} clean_acc={acc:.2f}")

        def forward_fn(edge_index_mod, x_arg, _data=data, _model=model):
            orig_ei = _data.edge_index; orig_x = _data.x
            _data.edge_index = edge_index_mod
            _data.x = x_arg
            try:
                out = _model(_data)
            finally:
                _data.edge_index = orig_ei
                _data.x = orig_x
            return out

        test_idx = tsm.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        rng = np.random.RandomState(rs)
        sample = rng.choice(test_idx, size=min(max_test, len(test_idx)), replace=False)
        sample = torch.tensor(sorted(sample.tolist()))

        for explainer in explainers:
            out = compute_node_fidelity(
                model=model, x=data.x, edge_index=base_ei, y=y,
                test_idx=sample, device=device, forward_fn=forward_fn,
                explainer=explainer, topk_list=topk_list,
                target="pred", K_hop=FLAGS.K_hop, seed=rs)
            for tk in topk_list:
                r = out["per_topk"][tk]
                if r.get("n_eval", 0) == 0:
                    continue
                agg[(explainer, tk)]["fp"].append(r["fid_plus_mean"])
                agg[(explainer, tk)]["fm"].append(r["fid_minus_mean"])
                agg[(explainer, tk)]["ch"].append(r["char_mean"])
                agg[(explainer, tk)]["n_eval"].append(r["n_eval"])
                print(f"[INTERP_NODE_V2_RAW] method=BGRL dataset={FLAGS.dataset} "
                      f"split=0 seed={rs} explainer={explainer} topk={tk} "
                      f"fid_plus={r['fid_plus_mean']:.4f} fid_minus={r['fid_minus_mean']:.4f} "
                      f"char={r['char_mean']:.4f} n_eval={r['n_eval']}")

    for (explainer, tk), d in agg.items():
        if not d["fp"]:
            continue
        fp = np.array(d["fp"]); fm = np.array(d["fm"]); ch = np.array(d["ch"])
        print(f"[INTERP_NODE_V2_AGG] method=BGRL dataset={FLAGS.dataset} "
              f"explainer={explainer} topk={tk} n_runs={len(d['fp'])} "
              f"fid_plus=\"{fp.mean():.4f} ± {fp.std(ddof=1) if len(fp)>1 else 0:.4f}\" "
              f"fid_minus=\"{fm.mean():.4f} ± {fm.std(ddof=1) if len(fm)>1 else 0:.4f}\" "
              f"char=\"{ch.mean():.4f} ± {ch.std(ddof=1) if len(ch)>1 else 0:.4f}\" "
              f"n_eval_mean={int(np.mean(d['n_eval']))}")
    print("=== Done ===")


if __name__ == "__main__":
    app.run(main)
