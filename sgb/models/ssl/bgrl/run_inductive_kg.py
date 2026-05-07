"""BGRL inductive KG link-prediction on FB15K237/WN18RR."""
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
from sgb.data.inductive_kg_runner import run_inductive_kg

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'WN18RR', 'KG TAG dataset.')
flags.DEFINE_string('ckpt_path', None, 'BGRL pretrained encoder.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 600, 'Max epochs.')
flags.DEFINE_integer('eval_every', 20, 'Eval every (epochs).')
flags.DEFINE_integer('patience', 10, 'Early stop patience.')
flags.DEFINE_float('lr', 2e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_integer('n_seeds', 3, 'Number of seeds.')


class _KGModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        from torch_geometric.data import Data
        d = Data(x=x, edge_index=edge_index)
        return self.encoder(d)


_CKPT = None


def _builder(in_channels, device):
    global _CKPT
    if _CKPT is None:
        ckpt = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            ckpt = ckpt['model']
        _CKPT = ckpt
    encoder = GCN([in_channels] + list(FLAGS.graph_encoder_layer), batchnorm=True)
    encoder.load_state_dict(_CKPT)
    return _KGModel(encoder).to(device)


def main(argv):
    run_inductive_kg(
        method_tag="BGRL_FT",
        dataset=FLAGS.dataset,
        build_ft_model=_builder,
        hidden_dim=FLAGS.graph_encoder_layer[-1],
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        eval_every=FLAGS.eval_every,
        patience=FLAGS.patience,
        n_seeds=FLAGS.n_seeds,
    )


if __name__ == "__main__":
    app.run(main)
