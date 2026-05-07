"""BGRL FT + step imbalance evaluation (node classification only).

Protocol: step imbalance following TAM/ReNode.
  - minor = |Y|/2 lowest-count classes
  - n_minor = n_major_max / rho
  - n_reps repetitions per rho

Outputs [IMB_RAW], [IMB_PER_CLASS], [IMB_AGG] lines.
"""

import copy
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from sklearn.metrics import balanced_accuracy_score

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN
from sgb.data.tag_registry import load as load_tag
from sgb.data.imbalance_splits import make_step_imbalance, compute_imbalance_metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'TAG dataset name.')
flags.DEFINE_string('ckpt_path', None, 'Pretrained BGRL encoder .pt.')
flags.DEFINE_integer('rho', 10, 'Imbalance ratio.')
flags.DEFINE_integer('n_reps', 10, 'Number of repetitions.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')


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
    m = torch.zeros(N, dtype=torch.bool)
    m[idx] = True
    return m


def _get_base_masks(data):
    if hasattr(data, 'train_masks') and data.train_masks is not None:
        return data.train_masks[0].bool(), data.val_masks[0].bool(), data.test_masks[0].bool()
    if hasattr(data, 'splits') and isinstance(data.splits, dict):
        s = data.splits
        N = data.num_nodes
        return (_idx_to_mask(s['train'], N),
                _idx_to_mask(s.get('valid', s.get('val')), N),
                _idx_to_mask(s['test'], N))
    tm, vm, tsm = data.train_mask, data.val_mask, data.test_mask
    if tm.dim() == 2:
        return tm[:, 0].bool(), vm[:, 0].bool(), (tsm[:, 0] if tsm.dim() == 2 else tsm).bool()
    return tm.bool(), vm.bool(), tsm.bool()


def train_ft(model, data, y, train_mask, val_mask, device):
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
    best_val_bacc, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        F.cross_entropy(logits[train_mask], y[train_mask]).backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(-1)
            val_pred = pred[val_mask].cpu().numpy()
            val_true = y[val_mask].cpu().numpy()
            val_bacc = balanced_accuracy_score(val_true, val_pred) * 100.0

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FLAGS.patience:
                break
    model.load_state_dict(best_state)
    return model


def main(argv):
    torch.manual_seed(42)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"[BGRL FT-IMB] Using {device}, dataset={FLAGS.dataset}, rho={FLAGS.rho}, n_reps={FLAGS.n_reps}")

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
    input_size = data.x.size(1)

    train_mask_base, val_mask, test_mask = _get_base_masks(data)
    train_mask_base = train_mask_base.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    ckpt_state = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and 'model' in ckpt_state:
        ckpt_state = ckpt_state['model']

    print(f"[BGRL FT-IMB] N={data.num_nodes}, d={input_size}, C={num_classes}")

    all_results = []
    for rep_idx in range(FLAGS.n_reps):
        seed = rep_idx
        torch.manual_seed(seed)
        np.random.seed(seed)

        imb_train_mask, meta = make_step_imbalance(
            train_mask=train_mask_base, y=y, rho=FLAGS.rho, seed=seed,
        )
        imb_train_mask = imb_train_mask.to(device)

        if rep_idx == 0:
            print(f"  [rep 0 meta] n_major_max={meta['n_major_max']} "
                  f"n_minor_target={meta['n_minor_target']} "
                  f"minor_classes={meta['minor_classes']}")

        encoder = GCN([input_size] + list(FLAGS.graph_encoder_layer), batchnorm=True)
        encoder.load_state_dict(ckpt_state)
        # Attach representation_size for our FTModel
        encoder.representation_size = FLAGS.graph_encoder_layer[-1]
        model = FTModel(encoder, num_classes, FLAGS.dropout).to(device)
        model = train_ft(model, data, y, imb_train_mask, val_mask, device)

        model.eval()
        with torch.no_grad():
            pred = model(data).argmax(-1)
        metrics = compute_imbalance_metrics(y[test_mask].cpu().numpy(),
                                            pred[test_mask].cpu().numpy(),
                                            num_classes=num_classes)
        all_results.append({"rep": rep_idx, **metrics})

        print(f"[IMB_RAW] method=BGRL_FT dataset={FLAGS.dataset} rho={FLAGS.rho} "
              f"rep={rep_idx} seed={seed} "
              f"bacc={metrics['bacc']:.4f} macro_f1={metrics['macro_f1']:.4f} "
              f"acc={metrics['acc']:.4f} n_minor_target={meta['n_minor_target']}")
        print(f"[IMB_PER_CLASS] method=BGRL_FT dataset={FLAGS.dataset} rho={FLAGS.rho} "
              f"rep={rep_idx} minor_classes={meta['minor_classes']} "
              f"per_class_recall={metrics['per_class_recall']} "
              f"per_class_f1={metrics['per_class_f1']}")

    bacc = np.array([r['bacc'] for r in all_results])
    f1 = np.array([r['macro_f1'] for r in all_results])
    acc = np.array([r['acc'] for r in all_results])
    print(f"\n=== BGRL Imbalance ({FLAGS.dataset}, rho={FLAGS.rho}, n={len(all_results)}) ===")
    print(f"  bAcc    = {bacc.mean():.2f} ± {bacc.std():.2f}")
    print(f"  macroF1 = {f1.mean():.2f} ± {f1.std():.2f}")
    print(f"  acc     = {acc.mean():.2f} ± {acc.std():.2f}")
    print(f"[IMB_AGG] method=BGRL_FT dataset={FLAGS.dataset} rho={FLAGS.rho} "
          f"n_reps={len(all_results)} "
          f"bacc=\"{bacc.mean():.2f} ± {bacc.std():.2f}\" "
          f"macro_f1=\"{f1.mean():.2f} ± {f1.std():.2f}\" "
          f"acc=\"{acc.mean():.2f} ± {acc.std():.2f}\"")


if __name__ == "__main__":
    app.run(main)
