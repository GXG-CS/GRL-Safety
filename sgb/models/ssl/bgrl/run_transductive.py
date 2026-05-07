"""BGRL fine-tuning (FT) eval on a single TAG node-classification dataset.

Loads a pretrained BGRL encoder (`encoder_ep*.pt` from train_joint.py),
attaches a linear classification head, and fine-tunes both end-to-end on
the downstream dataset with early stopping. Unlike `linear_eval_transductive.py`,
the encoder is NOT frozen.

This exists so BGRL can participate in the benchmark's unified FT protocol
alongside GIT / GFT / OFA / GraphMAE — BGRL's original paper reports LP, but
the encoder is a plain GCN and can be fine-tuned without any architectural
change.

Usage:
    python run_transductive.py \
        --dataset=cora \
        --graph_encoder_layer=768 --graph_encoder_layer=768 \
        --ckpt_path=ckpts/BGRL/joint/encoder_ep050.pt
"""
import copy
import logging
import os
import os.path as osp
import sys

from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BGRL_DIR = osp.dirname(osp.abspath(__file__))
if _BGRL_DIR not in sys.path:
    sys.path.insert(0, _BGRL_DIR)
_PROJECT_ROOT = osp.abspath(osp.join(_BGRL_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bgrl import GCN, get_dataset, get_wiki_cs  # type: ignore

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS

# --- Core flags -------------------------------------------------------------
flags.DEFINE_string('dataset', 'cora', 'TAG dataset name.')
flags.DEFINE_string('dataset_dir', './data', 'Unused for TAG; kept for API parity.')
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Encoder layer sizes.')
flags.DEFINE_string('ckpt_path', None, 'Path to pretrained BGRL encoder .pt.')

# --- Training flags ---------------------------------------------------------
flags.DEFINE_integer('max_epochs', 500, 'Max FT epochs per split.')
flags.DEFINE_integer('patience', 100, 'Early stop patience (epochs without val improvement).')
flags.DEFINE_float('lr', 1e-3, 'FT learning rate.')
flags.DEFINE_float('weight_decay', 5e-4, 'FT weight decay.')
flags.DEFINE_float('dropout', 0.5, 'Dropout on head input.')

# --- Mini-batch mode (for products-scale datasets) -------------------------
# bs=0 -> full-batch path (cora/pubmed/arxiv/wikics). bs>0 -> NeighborLoader
# mini-batch path, required for products-scale graphs where full-batch OOMs.
flags.DEFINE_integer('bs', 0, 'Batch size for NeighborLoader. 0 = full-batch.')
flags.DEFINE_integer('num_neighbors', 10, 'NeighborLoader num_neighbors per layer.')
flags.DEFINE_integer('eval_every', 1, 'Eval every N epochs (mini-batch mode; set >1 to speed up).')

TAG_NODE_DATASETS = {'cora', 'pubmed', 'wikics', 'arxiv'}


class FTModel(nn.Module):
    """Pretrained encoder + linear classification head, trained end-to-end."""

    def __init__(self, encoder: GCN, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.representation_size, num_classes)

    def forward(self, data):
        h = self.encoder(data)
        h = self.dropout(h)
        return self.head(h)


def _build_fresh_ft_model(ckpt_state, num_classes, input_size, device):
    """Build a fresh GCN + head, load pretrained encoder weights. Called per split."""
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)
    encoder.load_state_dict(ckpt_state)
    model = FTModel(encoder, num_classes=num_classes, dropout=FLAGS.dropout).to(device)
    return model


def _split_iter(data):
    """Yield (train_mask, val_mask, test_mask) per split for any TAG node ds.

    - cora/pubmed ship a `list` of 10 per-split masks under train_masks/val_masks/test_masks.
    - wikics ships `train_mask/val_mask` as [N, 20] bool tensors and shares `test_mask`.
    - arxiv ships single-split `train_mask/val_mask/test_mask` as [N] bool.
    """
    if hasattr(data, 'train_masks') and isinstance(data.train_masks, list):
        for tr, va, te in zip(data.train_masks, data.val_masks, data.test_masks):
            yield tr.bool(), va.bool(), te.bool()
        return

    if hasattr(data, 'train_mask'):
        tm, vm, testm = data.train_mask, data.val_mask, data.test_mask
        if tm.dim() == 2:
            # wikics: [N, 20] for train/val, [N] for test
            if testm.dim() == 2:
                testm = testm[:, 0]
            for s in range(tm.shape[1]):
                yield tm[:, s].bool(), vm[:, s].bool(), testm.bool()
            return
        # arxiv: single split
        yield tm.bool(), vm.bool(), testm.bool()
        return

    raise RuntimeError(f"No usable splits on {FLAGS.dataset}")


def _train_one_split(model, data, y, train_mask, val_mask, test_mask, device):
    """Dispatch: full-batch (FLAGS.bs=0) vs NeighborLoader mini-batch (FLAGS.bs>0)."""
    if FLAGS.bs > 0:
        return _train_one_split_minibatch(model, data, y, train_mask, val_mask, test_mask, device)
    return _train_one_split_fullbatch(model, data, y, train_mask, val_mask, test_mask, device)


def _train_one_split_fullbatch(model, data, y, train_mask, val_mask, test_mask, device):
    """Full-batch FT. Used for cora/pubmed/arxiv/wikics (fits in GPU)."""
    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            logits = model(data)
            pred = logits.argmax(dim=-1)
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
            test_acc = (pred[test_mask] == y[test_mask]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= FLAGS.patience:
                break

    return best_test_acc, best_val_acc, best_epoch


def _mask_to_idx(mask):
    return mask.nonzero(as_tuple=False).view(-1)


def _eval_loader(model, loader, device):
    """Forward the loader, collect seed-node predictions, return accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            bs_seed = batch.input_id.shape[0]
            logits = model(batch)
            pred = logits[:bs_seed].argmax(dim=-1)
            y_seed = batch.y[:bs_seed]
            if y_seed.dim() > 1:
                y_seed = y_seed.squeeze()
            correct += (pred == y_seed).sum().item()
            total += bs_seed
    return correct / max(total, 1)


def _train_one_split_minibatch(model, data, y, train_mask, val_mask, test_mask, device):
    """NeighborLoader-based FT. Used for products-scale graphs where full-batch OOMs.

    data must already live on CPU here — loader moves each subgraph batch to
    GPU individually. (For full-batch mode we put the whole graph on GPU once.)
    """
    # Lazy import so full-batch path doesnt require torch_geometric.loader.
    from torch_geometric.loader import NeighborLoader

    # NeighborLoader wants index tensors, not bool masks.
    train_idx = _mask_to_idx(train_mask.cpu())
    val_idx = _mask_to_idx(val_mask.cpu())
    test_idx = _mask_to_idx(test_mask.cpu())

    # data is on device for full-batch; for mini-batch we need it back on CPU
    # so NeighborLoader can sample without materializing the whole subgraph in
    # GPU memory up front.
    data_cpu = data.cpu()

    # NeighborLoader cannot slice list-typed attributes (raw_texts,
    # label_names, etc.) — strip them. Keep only tensor attributes that are
    # either node-level (for the subgraph forward) or edge-level.
    from torch_geometric.data import Data as _PygData
    keep = {}
    for k, v in data_cpu.to_dict().items():
        if isinstance(v, torch.Tensor):
            keep[k] = v
    data_cpu = _PygData(**keep)

    num_layers = len(FLAGS.graph_encoder_layer)
    common_kw = dict(num_neighbors=[FLAGS.num_neighbors] * num_layers,
                     batch_size=FLAGS.bs, num_workers=0)
    train_loader = NeighborLoader(data_cpu, input_nodes=train_idx, shuffle=True, **common_kw)
    val_loader = NeighborLoader(data_cpu, input_nodes=val_idx, shuffle=False, **common_kw)
    test_loader = NeighborLoader(data_cpu, input_nodes=test_idx, shuffle=False, **common_kw)

    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    best_val_acc = -1.0
    best_test_acc = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, FLAGS.max_epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_steps = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            bs_seed = batch.input_id.shape[0]
            optim.zero_grad()
            logits = model(batch)
            y_seed = batch.y[:bs_seed]
            if y_seed.dim() > 1:
                y_seed = y_seed.squeeze()
            loss = F.cross_entropy(logits[:bs_seed], y_seed)
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            ep_steps += 1

        if epoch % FLAGS.eval_every != 0 and epoch != FLAGS.max_epochs:
            continue

        val_acc = _eval_loader(model, val_loader, device)
        test_acc = _eval_loader(model, test_loader, device)
        print(f"    epoch {epoch:4d}  loss={ep_loss/max(ep_steps,1):.4f}  "
              f"val={val_acc*100:.2f}  test={test_acc*100:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += FLAGS.eval_every
            if no_improve >= FLAGS.patience:
                break

    return best_test_acc, best_val_acc, best_epoch


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using %s for FT evaluation.', device)

    # --- Data ---
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    else:
        dataset, _, _, _ = get_wiki_cs(FLAGS.dataset_dir)

    data = dataset[0]
    log.info('Dataset %s: %s', FLAGS.dataset, data)

    if data.y.dim() > 1:
        data.y = data.y.squeeze()
    # Full-batch: move whole graph to GPU once.
    # Mini-batch: keep graph on CPU, NeighborLoader moves per-batch subgraphs.
    if FLAGS.bs == 0:
        data = data.to(device)
    y = data.y
    num_classes = int(y.max().item() + 1)
    input_size = int(data.x.size(1))
    print(f"[BGRL FT] dataset={FLAGS.dataset}, N={data.num_nodes}, "
          f"input_dim={input_size}, num_classes={num_classes}, "
          f"mode={'minibatch(bs=%d)' % FLAGS.bs if FLAGS.bs > 0 else 'full-batch'}")

    # --- Pretrained encoder state_dict ---
    assert FLAGS.ckpt_path, "--ckpt_path is required."
    ckpt_state = torch.load(FLAGS.ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt_state, dict) and 'model' in ckpt_state:
        ckpt_state = ckpt_state['model']
    print(f"[BGRL FT] loaded pretrained encoder from {FLAGS.ckpt_path}")

    # --- Fine-tune across splits ---
    test_accs = []
    val_accs = []
    for split_idx, (tr, va, te) in enumerate(_split_iter(data)):
        torch.manual_seed(split_idx)
        np.random.seed(split_idx)

        model = _build_fresh_ft_model(ckpt_state, num_classes, input_size, device)
        test_acc, val_acc, best_epoch = _train_one_split(
            model, data, y, tr.to(device), va.to(device), te.to(device), device,
        )
        test_accs.append(test_acc)
        val_accs.append(val_acc)
        print(f"  [split {split_idx}] best_epoch={best_epoch}  "
              f"val={val_acc*100:.2f}  test={test_acc*100:.2f}")

    test_accs = np.array(test_accs)
    val_accs = np.array(val_accs)

    print("\n=== BGRL FT Result ===")
    print(f"Ckpt:    {FLAGS.ckpt_path}")
    print(f"Dataset: {FLAGS.dataset}  (n_splits={len(test_accs)})")
    print(f"Val:     {val_accs.mean()*100:.2f} +/- {val_accs.std()*100:.2f}")
    print(f"Test:    {test_accs.mean()*100:.2f} +/- {test_accs.std()*100:.2f}")


if __name__ == '__main__':
    log.info('PyTorch version: %s', torch.__version__)
    app.run(main)
