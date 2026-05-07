"""BGRL FT + structural (degree-based) fairness."""

import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _BGRL_DIR not in sys.path: sys.path.insert(0, _BGRL_DIR)
if _PROJECT_ROOT not in sys.path: sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN
from sgb.data.tag_registry import load as load_tag
from sgb.metrics.fairness import compute_structural_fairness

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'tolokers', 'Dataset.')
flags.DEFINE_string('ckpt_path', None, 'Pretrained BGRL encoder .pt.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')
flags.DEFINE_float('q', 0.2, 'Quantile.')
flags.DEFINE_bool('debug', False, 'smoke.')
flags.DEFINE_integer('num_splits', 5, 'Number of split seeds (clamp <=5).')
flags.DEFINE_integer('num_seeds', 5, 'Number of run seeds (clamp <=5).')

SPLIT_SEEDS = [0, 1, 2, 3, 4]
RUN_SEEDS = [42, 43, 44, 45, 46]


class FTModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)
    def forward(self, data):
        h = self.encoder(data); h = self.dropout(h); return self.head(h)


def _idx_to_mask(idx, N):
    m = torch.zeros(N, dtype=torch.bool); m[idx] = True; return m


def _get_splits(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        avail = len(data.train_masks)
        return [(data.train_masks[i % avail].bool(),
                 data.val_masks[i % avail].bool(),
                 data.test_masks[i % avail].bool()) for i in range(5)]
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits; N = data.num_nodes
        tm = _idx_to_mask(s['train'], N); vm = _idx_to_mask(s.get('valid', s.get('val')), N); tsm = _idx_to_mask(s['test'], N)
        return [(tm, vm, tsm)] * 5
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        n = min(5, tm.size(1))
        return [(tm[:, i].bool(), vm[:, i].bool(),
                 (tsm[:, i] if tsm.dim() == 2 else tsm).bool()) for i in range(n)]
    return [(tm.bool(), vm.bool(), tsm.bool())] * 5


def train_ft(model, data, y, tm, vm, device):
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    best_val, best_state, no_imp = -1.0, None, 0
    for _ in range(1, FLAGS.max_epochs + 1):
        model.train(); optim.zero_grad()
        F.cross_entropy(model(data)[tm], y[tm]).backward()
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[BGRL FT-STRUCT] dataset={FLAGS.dataset}")

    data, _ = load_tag(FLAGS.dataset)
    if data.x is None:
        data.x = data.node_text_feat
    elif data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat
    if data.y.dim() > 1: data.y = data.y.squeeze()
    data = data.to(device)
    y = data.y
    num_classes = int(y.max().item()) + 1
    input_size = data.x.size(1)
    N = int(data.num_nodes)

    degree = torch.bincount(data.edge_index[0], minlength=N).cpu().numpy()
    print(f"[BGRL FT-STRUCT] N={N} d={input_size} C={num_classes} deg mean={degree.mean():.1f}")

    ckpt_state = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and 'model' in ckpt_state:
        ckpt_state = ckpt_state['model']

    splits = _get_splits(data)
    n_split = min(FLAGS.num_splits, 5)
    n_seed = min(FLAGS.num_seeds, 5)
    split_seeds = SPLIT_SEEDS[:1] if FLAGS.debug else SPLIT_SEEDS[:n_split]
    run_seeds = RUN_SEEDS[:1] if FLAGS.debug else RUN_SEEDS[:n_seed]

    results = []
    for split_idx, (tm, vm, tsm) in enumerate(splits):
        if split_idx >= len(split_seeds): break
        tm = tm.to(device); vm = vm.to(device); tsm = tsm.to(device)
        for rs in run_seeds:
            torch.manual_seed(rs); np.random.seed(rs)
            encoder = GCN([input_size] + list(FLAGS.graph_encoder_layer), batchnorm=True)
            encoder.load_state_dict(ckpt_state)
            encoder.representation_size = FLAGS.graph_encoder_layer[-1]
            model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)
            model = train_ft(model, data, y, tm, vm, device)
            model.eval()
            with torch.no_grad():
                preds = model(data).argmax(-1).cpu().numpy()
                y_np = y.cpu().numpy()
            tsm_np = tsm.cpu().numpy()
            acc = (y_np[tsm_np] == preds[tsm_np]).mean() * 100.0
            struct = compute_structural_fairness(y_np, preds, degree, test_mask=tsm_np, q=FLAGS.q)
            print(f"[STRUCT_RAW] method=BGRL_FT dataset={FLAGS.dataset} split={split_idx} seed={rs} "
                  f"acc={acc:.4f} acc_head={struct['acc_head']:.4f} acc_tail={struct['acc_tail']:.4f} "
                  f"acc_gap={struct['acc_gap']:.4f} f1_head={struct['f1_head']:.4f} f1_tail={struct['f1_tail']:.4f} "
                  f"f1_gap={struct['f1_gap']:.4f} n_head={struct['n_head']} n_tail={struct['n_tail']} q={FLAGS.q}")
            results.append({'acc': acc, **struct})

    def _agg(k):
        vs = [r[k] for r in results
              if r[k] is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
        if not vs: return float('nan'), float('nan')
        return float(np.mean(vs)), float(np.std(vs))
    a, sa = _agg('acc'); ah, sah = _agg('acc_head'); at, sat = _agg('acc_tail'); ag, sag = _agg('acc_gap')
    fh, sfh = _agg('f1_head'); ft, sft = _agg('f1_tail'); fg, sfg = _agg('f1_gap')
    print(f'[STRUCT_AGG] method=BGRL_FT dataset={FLAGS.dataset} n_runs={len(results)} '
          f'acc="{a:.2f} ± {sa:.2f}" acc_head="{ah:.2f} ± {sah:.2f}" acc_tail="{at:.2f} ± {sat:.2f}" '
          f'acc_gap="{ag:.2f} ± {sag:.2f}" f1_head="{fh:.2f} ± {sfh:.2f}" f1_tail="{ft:.2f} ± {sft:.2f}" '
          f'f1_gap="{fg:.2f} ± {sfg:.2f}" q={FLAGS.q}')


if __name__ == "__main__":
    app.run(main)
