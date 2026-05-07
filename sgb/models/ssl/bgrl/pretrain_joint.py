"""BGRL joint pretraining on GFT's 9 TAG datasets.

Mirrors GFT's joint-pretrain recipe:
  - disjoint-union of cora/pubmed/arxiv/wikics/WN18RR/FB15K237/chemhiv/chemblpre/chempcba
  - per-epoch weighted node sampling (from sgb/models/gfm/gft/config/pt_data.yaml "all")
  - NeighborLoader mini-batches with num_neighbors=[10]*num_layers
  - SBERT 768d features, materialized lazily per batch

Adapts BGRLs native per-node cosine loss to the mini-batch setting:
  - loss is computed only over the `input_id` (seed) nodes of each batch
  - per-step lr/momentum schedule instead of per-epoch

Unchanged from BGRLs original design:
  - online/target encoder architecture
  - MLP predictor
  - EMA target network update
  - edge dropout + feature dropout augmentations
"""
import copy
import logging
import os
import time

from absl import app
from absl import flags
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bgrl import (
    BGRL,
    GCN,
    MLP_Predictor,
    CosineDecayScheduler,
    get_graph_drop_transform,
    set_random_seeds,
)
from bgrl.data import get_joint_pretrain_data, get_joint_train_nodes

log = logging.getLogger(__name__)


def _find_latest_ckpt(logdir):
    """Return path to highest-epoch ckpt_ep*.pt in logdir, or None."""
    if not os.path.isdir(logdir):
        return None
    best_epoch = -1
    best_path = None
    for fn in os.listdir(logdir):
        if fn.startswith('ckpt_ep') and fn.endswith('.pt'):
            try:
                n = int(fn[len('ckpt_ep'):-len('.pt')])
            except ValueError:
                continue
            if n > best_epoch:
                best_epoch = n
                best_path = os.path.join(logdir, fn)
    return best_path


def _save_checkpoint(path, *, epoch, global_step, model, optimizer, rng_state):
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state': model.state_dict(),  # online + target + predictor
        'optimizer_state': optimizer.state_dict(),
        'rng_state': rng_state,
    }, path)


def _load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if ckpt.get('rng_state') is not None:
        torch.set_rng_state(ckpt['rng_state'].cpu())
    return ckpt['epoch'], ckpt['global_step']


FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', 0, 'Random seed.')

# Data / joint setting
flags.DEFINE_string('weight_setting', 'all', 'Which weight set to use from JOINT_PRETRAIN_WEIGHTS.')

# Architecture (matches our benchmark decision: BGRL 2 layers, 768d)
flags.DEFINE_multi_integer('graph_encoder_layer', [768, 768], 'Conv layer sizes.')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training
flags.DEFINE_integer('epochs', 50, 'Number of pretraining epochs.')
flags.DEFINE_integer('batch_size', 1024, 'NeighborLoader batch size (seed nodes per step).')
flags.DEFINE_integer('num_neighbors', 10, 'Number of neighbors per GNN layer for sampling.')
flags.DEFINE_float('lr', 1e-4, 'Peak learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('mm', 0.99, 'Base target network EMA momentum.')
flags.DEFINE_float('warmup_frac', 0.1, 'Fraction of total steps used for lr warmup.')

# Augmentations
flags.DEFINE_float('drop_edge_p_1', 0.2, 'Edge dropout prob, view 1.')
flags.DEFINE_float('drop_feat_p_1', 0.2, 'Feature dropout prob, view 1.')
flags.DEFINE_float('drop_edge_p_2', 0.2, 'Edge dropout prob, view 2.')
flags.DEFINE_float('drop_feat_p_2', 0.2, 'Feature dropout prob, view 2.')

# Logging / checkpointing
flags.DEFINE_string('logdir', None, 'Where checkpoints and TB logs go.')
flags.DEFINE_integer('log_steps', 50, 'Log to TB every N steps.')
flags.DEFINE_integer('max_steps_per_epoch', 0, 'If >0, cap steps per epoch (for smoke test).')
flags.DEFINE_string('resume_from', '',
                    'Path to a ckpt_ep*.pt file to resume from, or "latest" to '
                    'auto-pick the highest-epoch file in --logdir. Empty = start fresh.')
flags.DEFINE_integer('ckpt_every', 1, 'Save a full checkpoint every N epochs.')


def main(argv):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using %s for training.', device)

    set_random_seeds(random_seed=FLAGS.model_seed)

    assert FLAGS.logdir, '--logdir is required.'
    os.makedirs(FLAGS.logdir, exist_ok=True)
    with open(os.path.join(FLAGS.logdir, 'config.cfg'), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # ---- Data ----
    print(f'[BGRL-joint] Building disjoint-union of 9 datasets '
          f'(weight_setting={FLAGS.weight_setting})...')
    t0 = time.time()
    big_data, node_text_feat, edge_text_feat, dataset_names, weights = \
        get_joint_pretrain_data(FLAGS.weight_setting)
    print(f'[BGRL-joint] big_data: {big_data.num_nodes} nodes, '
          f'{big_data.edge_index.shape[1]} edges '
          f'(built in {time.time()-t0:.1f}s)')
    print(f'[BGRL-joint] node_text_feat: {tuple(node_text_feat.shape)} '
          f'({node_text_feat.numel()*4/1e6:.0f} MB)')
    print(f'[BGRL-joint] datasets: {dataset_names}')

    # Feature table lives on GPU so each batchs lookup is a GPU gather.
    node_text_feat = node_text_feat.to(device)

    # ---- Model ----
    input_size = node_text_feat.shape[1]  # 768
    representation_size = FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)
    predictor = MLP_Predictor(representation_size, representation_size,
                              hidden_size=FLAGS.predictor_hidden_size)
    model = BGRL(encoder, predictor).to(device)

    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr,
                      weight_decay=FLAGS.weight_decay)

    # ---- Augmentations ----
    transform_1 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_1,
                                           drop_feat_p=FLAGS.drop_feat_p_1)
    transform_2 = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p_2,
                                           drop_feat_p=FLAGS.drop_feat_p_2)

    writer = SummaryWriter(FLAGS.logdir)

    # ---- Compute total steps for the per-step schedule ----
    # We rebuild train_nodes each epoch, but their count is deterministic
    # from the weight table — compute it up front for the schedule.
    train_nodes_probe = get_joint_train_nodes(big_data, dataset_names, weights)
    steps_per_epoch = (train_nodes_probe.shape[0] + FLAGS.batch_size - 1) // FLAGS.batch_size
    if FLAGS.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, FLAGS.max_steps_per_epoch)
    total_steps = steps_per_epoch * FLAGS.epochs
    warmup_steps = max(1, int(FLAGS.warmup_frac * total_steps))
    print(f'[BGRL-joint] steps_per_epoch={steps_per_epoch}, '
          f'total_steps={total_steps}, warmup_steps={warmup_steps}')

    lr_scheduler = CosineDecayScheduler(FLAGS.lr, warmup_steps, total_steps)
    mm_scheduler = CosineDecayScheduler(1 - FLAGS.mm, 0, total_steps)

    # ---- Resume (optional) ----
    start_epoch = 0
    global_step = 0
    resume_path = None
    if FLAGS.resume_from == 'latest':
        resume_path = _find_latest_ckpt(FLAGS.logdir)
        if resume_path is None:
            print(f'[BGRL-joint] --resume_from=latest but no ckpt in {FLAGS.logdir}; starting fresh.')
    elif FLAGS.resume_from:
        resume_path = FLAGS.resume_from

    if resume_path is not None:
        print(f'[BGRL-joint] resuming from {resume_path}')
        start_epoch, global_step = _load_checkpoint(resume_path, model, optimizer, device)
        print(f'[BGRL-joint] resumed at epoch={start_epoch}, global_step={global_step}')
        if start_epoch >= FLAGS.epochs:
            print(f'[BGRL-joint] start_epoch {start_epoch} >= target epochs {FLAGS.epochs}; nothing to do.')
            return

    # ---- Training ----
    for epoch in range(start_epoch + 1, FLAGS.epochs + 1):
        train_nodes = get_joint_train_nodes(big_data, dataset_names, weights)
        loader = NeighborLoader(
            big_data,
            input_nodes=train_nodes,
            num_neighbors=[FLAGS.num_neighbors] * len(FLAGS.graph_encoder_layer),
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=0,
        )

        model.train()
        pbar = tqdm(loader, desc=f'epoch {epoch}/{FLAGS.epochs}', leave=False)
        for local_step, batch in enumerate(pbar):
            if FLAGS.max_steps_per_epoch > 0 and local_step >= FLAGS.max_steps_per_epoch:
                break

            # Per-step lr / momentum
            cur_lr = lr_scheduler.get(global_step)
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr
            cur_mm = 1 - mm_scheduler.get(global_step)

            # Materialize features on GPU.
            batch = batch.to(device, non_blocking=True)
            batch.x = node_text_feat[batch.x]
            # Seed nodes live at positions [0, bs_seed) in the subgraph.
            bs_seed = batch.input_id.shape[0]

            # Two augmented views.
            x1 = transform_1(batch)
            x2 = transform_2(batch)

            q1, y2 = model(x1, x2)
            q2, y1 = model(x2, x1)

            # Loss only over seed nodes.
            loss = (
                2
                - cosine_similarity(q1[:bs_seed], y2[:bs_seed].detach(), dim=-1).mean()
                - cosine_similarity(q2[:bs_seed], y1[:bs_seed].detach(), dim=-1).mean()
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_target_network(cur_mm)

            if global_step % FLAGS.log_steps == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('params/lr', cur_lr, global_step)
                writer.add_scalar('params/mm', cur_mm, global_step)
                pbar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{cur_lr:.2e}')

            global_step += 1

        # Save a numbered encoder for downstream eval every epoch, and a
        # full training-state ckpt every `ckpt_every` epochs (for resume).
        encoder_path = os.path.join(FLAGS.logdir, f'encoder_ep{epoch:03d}.pt')
        torch.save(model.online_encoder.state_dict(), encoder_path)
        # Also mirror to encoder.pt so downstream code with a fixed path
        # always sees the most recent encoder.
        torch.save(model.online_encoder.state_dict(),
                   os.path.join(FLAGS.logdir, 'encoder.pt'))

        if epoch % FLAGS.ckpt_every == 0 or epoch == FLAGS.epochs:
            ckpt_path = os.path.join(FLAGS.logdir, f'ckpt_ep{epoch:03d}.pt')
            _save_checkpoint(
                ckpt_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                rng_state=torch.get_rng_state(),
            )
            print(f'[BGRL-joint] epoch {epoch} done, saved {encoder_path} + {ckpt_path}')
        else:
            print(f'[BGRL-joint] epoch {epoch} done, saved {encoder_path}')

    writer.close()
    print('[BGRL-joint] training complete.')


if __name__ == '__main__':
    app.run(main)
