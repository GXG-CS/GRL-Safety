"""GNN baseline (GCN/GAT/SAGE) FT + graph-level interpretation eval.

Mirror of run_scaffold_ood.py but calls run_graph_interpretation.
Supported datasets: bbbp, bace (scaffold split), mutag (random split).
"""
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
from sgb.data.graph_interpretation_runner import run_graph_interpretation

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', None, 'bbbp, bace, or mutag.')
flags.DEFINE_string('model', 'gcn', 'gcn, gat, or sage.')
flags.DEFINE_integer('hidden', 768, 'Hidden dim.')
flags.DEFINE_integer('num_layers', 2, 'Encoder layers.')
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs.')
flags.DEFINE_integer('patience', 200, 'Early stop patience.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay.')
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


def _builder(in_channels, num_tasks, dropout, device):
    enc = GNNEncoderWrapper(
        model_name=FLAGS.model,
        in_channels=in_channels,
        hidden_channels=FLAGS.hidden,
        num_layers=FLAGS.num_layers,
        dropout=FLAGS.dropout,
    )
    return _FTGraphModel(enc, num_tasks, dropout).to(device)


def main(argv):
    motif_masks = None
    if FLAGS.dataset == "mutag":
        try:
            from sgb.metrics.mutag_motif import load_mutag_motif_masks
            motif_masks = load_mutag_motif_masks()
        except Exception as e:
            print(f"[motif] WARN: motif loader failed: {e}; running without motif eval")

    run_graph_interpretation(
        method_tag=METHOD_NAMES[FLAGS.model.lower()],
        dataset=FLAGS.dataset,
        build_ft_model=_builder,
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
        dropout=FLAGS.dropout,
        max_epochs=FLAGS.max_epochs,
        patience=FLAGS.patience,
        n_seeds=FLAGS.n_seeds,
        motif_masks=motif_masks,
    )


if __name__ == "__main__":
    app.run(main)
