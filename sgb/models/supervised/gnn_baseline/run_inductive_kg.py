"""GNN baseline (GCN/GAT/SAGE) inductive KG link-prediction on FB15K237/WN18RR."""
import os.path as osp
import sys

import torch
import torch.nn as nn
from absl import app, flags

_BASE_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_BASE_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from sgb.models.supervised.gnn_baseline import GNNEncoderWrapper, METHOD_NAMES
from sgb.data.inductive_kg_runner import run_inductive_kg

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'FB15K237', 'KG TAG dataset.')
flags.DEFINE_string('model', 'gcn', 'gcn/gat/sage.')
flags.DEFINE_integer('hidden', 768, 'Hidden dim.')
flags.DEFINE_integer('num_layers', 2, 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 100, 'Max epochs.')
flags.DEFINE_integer('eval_every', 5, 'Eval frequency (epochs).')
flags.DEFINE_integer('patience', 5, 'Early stop patience (eval rounds).')
flags.DEFINE_float('lr', 5e-4, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('dropout', 0.2, 'Dropout.')
flags.DEFINE_integer('n_negatives', 10, 'Negatives per positive (head+tail).')
flags.DEFINE_integer('n_seeds', 3, 'Number of partition seeds.')


class _KGModel(nn.Module):
    def __init__(self, model_name, in_channels, hidden, num_layers, dropout):
        super().__init__()
        self.encoder = GNNEncoderWrapper(
            model_name=model_name, in_channels=in_channels,
            hidden_channels=hidden, num_layers=num_layers, dropout=dropout,
        )

    def encode(self, x, edge_index):
        return self.encoder.encoder(x, edge_index)


def _builder(in_channels, device):
    return _KGModel(FLAGS.model, in_channels, FLAGS.hidden,
                    FLAGS.num_layers, FLAGS.dropout).to(device)


def main(argv):
    run_inductive_kg(
        method_tag=METHOD_NAMES[FLAGS.model.lower()],
        dataset=FLAGS.dataset,
        build_ft_model=_builder,
        hidden_dim=FLAGS.hidden,
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        max_epochs=FLAGS.max_epochs,
        eval_every=FLAGS.eval_every,
        patience=FLAGS.patience,
        n_negatives=FLAGS.n_negatives,
        n_seeds=FLAGS.n_seeds,
    )


if __name__ == "__main__":
    app.run(main)
