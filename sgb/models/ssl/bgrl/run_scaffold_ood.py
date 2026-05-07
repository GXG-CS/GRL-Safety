"""BGRL FT + scaffold-OOD eval on BBBP / BACE."""
import os.path as osp
import sys

import torch
import torch.nn as nn
from absl import app, flags

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN
from sgb.data.scaffold_ood_runner import run_scaffold_ood

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'bbbp or bace.')
flags.DEFINE_string('ckpt_path', None, 'Pretrained BGRL encoder .pt.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 1000, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')
flags.DEFINE_integer('n_seeds', 5, 'Number of FT seeds.')


class _FTGraphModel(nn.Module):
    def __init__(self, encoder, num_classes, dropout):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        from torch_geometric.nn import global_mean_pool
        h = self.encoder(data)
        h = global_mean_pool(h, data.batch)
        h = self.dropout(h)
        return self.head(h)


_CKPT_STATE = None


def _builder(in_channels, num_tasks, dropout, device):
    global _CKPT_STATE
    if _CKPT_STATE is None:
        ckpt = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']
        _CKPT_STATE = ckpt
    encoder = GCN([in_channels] + list(FLAGS.graph_encoder_layer), batchnorm=True)
    encoder.load_state_dict(_CKPT_STATE)
    return _FTGraphModel(encoder, num_tasks, dropout).to(device)


def main(argv):
    run_scaffold_ood(
        method_tag="BGRL_FT",
        dataset=FLAGS.dataset,
        build_ft_model=_builder,
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        dropout=FLAGS.dropout,
        max_epochs=FLAGS.max_epochs,
        patience=FLAGS.patience,
        n_seeds=FLAGS.n_seeds,
    )


if __name__ == "__main__":
    app.run(main)
