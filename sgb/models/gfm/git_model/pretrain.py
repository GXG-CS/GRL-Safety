import os
import os.path as osp
import sys
import random
import yaml
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling, mask_feature, dropout_adj
from torch_geometric.loader import NeighborLoader

from data.pretrain_data import (
    VirtualNodeAugmentor,
    postprocess,
    is_pretrain_group,
    unified_data_tag,
    build_weighted_train_nodes,
)
from model.encoder import Encoder, InnerProductDecoder
from model.pretrain_model import PretrainModel
from utils.utils import seed_everything, get_scheduler, get_device_from_model, check_path
from utils.args import get_args_pretrain
from utils.loader import get_pt_loader

import wandb

get_loader = get_pt_loader


# ---------------------------------------------------------------------- #
#  Checkpoint helpers (GFM-Safety: resume support)                        #
# ---------------------------------------------------------------------- #

_STATE_LATEST = "state_latest.pt"


def save_resume_state(path, epoch, model, optimizer, scheduler):
    """Atomically write a full training-state checkpoint for resume.

    Stores everything needed to restart mid-training: the full PretrainModel
    (not just the encoder), optimizer, scheduler, and the last completed
    epoch number. Writes to a .tmp file then os.replace() so a crash during
    save can't corrupt the previous checkpoint.
    """
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_resume_state(path, model, optimizer, scheduler, device):
    """Load a resume checkpoint written by save_resume_state.

    Returns the last completed epoch number (so the caller should start
    training from `last_epoch + 1`). Silently tolerates missing scheduler
    state, and restores RNG state best-effort.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            print("  [resume] WARN scheduler restore failed: {}".format(e))
    try:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    except Exception as e:
        print("  [resume] WARN RNG restore failed: {}".format(e))
    return int(ckpt["epoch"])


def pretrain(model, loader, optimizer, scheduler=None, **kwargs):
    model.train()
    device = get_device_from_model(model)
    params = kwargs['params']
    epoch_sums = {'feat': 0.0, 'topo': 0.0, 'sem': 0.0, 'align': 0.0, 'total': 0.0}
    num_batches = 0

    for data in loader:
        bs = data.batch_size

        x = data.node_text_feat[data.x].to(device)
        edge_index = data.edge_index.to(device)
        graph = [x, edge_index]

        x1, _ = mask_feature(x, p=params["feat_p"])
        edge_index1, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph1 = [x1, edge_index1]

        x2, _ = mask_feature(x, p=params["feat_p"])
        edge_index2, _ = dropout_adj(edge_index, p=params["edge_p"], force_undirected=True, num_nodes=x.size(0))
        aug_graph2 = [x2, edge_index2]

        losses = model(graph, aug_graph1, aug_graph2, bs=bs, params=params)
        loss = losses['loss']

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        wandb.log(
            {
                "loss/feat_loss": losses["feat_loss"].item(),
                "loss/topo_loss": losses["topo_loss"].item(),
                "loss/sem_loss": losses["sem_loss"].item(),
                "loss/align_reg": losses["align_reg"].item(),
                "loss/loss": loss.item(),
            }
        )
        epoch_sums['feat'] += losses["feat_loss"].item()
        epoch_sums['topo'] += losses["topo_loss"].item()
        epoch_sums['sem'] += losses["sem_loss"].item()
        epoch_sums['align'] += losses["align_reg"].item()
        epoch_sums['total'] += loss.item()
        num_batches += 1

    if num_batches == 0:
        return None
    return {k: v / num_batches for k, v in epoch_sums.items()}


def run(params):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(params["seed"])
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU

    # Make sgb.data.tag_registry importable from the project root.
    _project_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    ds_name = params["pretrain_dataset"]

    # ------------------------------------------------------------------ #
    #  Multi-dataset path (GFT-style): triggered when --pretrain_dataset   #
    #  names a group defined in git_model/config/pt_data.yaml.             #
    # ------------------------------------------------------------------ #
    use_group = is_pretrain_group(ds_name)

    if use_group:
        pretrain_data, task_node_idx_dict, weights_dict = unified_data_tag(ds_name)
        task_node_idx = None  # not used directly in the group path
        print(
            "[GIT] Unified batch: num_nodes={}, num_edges={}, node_text_feat={}"
            .format(pretrain_data.num_nodes,
                    pretrain_data.edge_index.size(1),
                    pretrain_data.node_text_feat.size(0))
        )
    else:
        # Single-dataset fallback — original behavior.
        from sgb.data.tag_registry import load as load_tag
        pretrain_data, slices = load_tag(ds_name)

        # Molecule data: rename node_embs -> node_text_feat (same as GFT)
        if hasattr(pretrain_data, 'node_embs') and pretrain_data.node_embs is not None:
            pretrain_data.node_text_feat = pretrain_data.node_embs
            pretrain_data.node_embs = None
        if hasattr(pretrain_data, 'edge_embs') and pretrain_data.edge_embs is not None:
            pretrain_data.edge_text_feat = pretrain_data.edge_embs
            pretrain_data.edge_embs = None
        if hasattr(pretrain_data, 'pretrain_edge_index') and pretrain_data.pretrain_edge_index is not None:
            pretrain_data.edge_index = pretrain_data.pretrain_edge_index
            pretrain_data.pretrain_edge_index = None

        if pretrain_data.x.ndim == 2:
            pretrain_data.x = torch.arange(pretrain_data.node_text_feat.size(0), dtype=torch.long)

        # Determine task type from dataset
        from data.pretrain_data import dataset2domain, domain2task
        task = domain2task.get(dataset2domain.get(ds_name, 'citation'), 'node')

        # Graph classification: reconstruct groups (node-to-graph mapping) from slices
        if task == 'graph' and not hasattr(pretrain_data, 'groups'):
            if slices is not None and 'x' in slices:
                ptr = slices['x']  # [num_graphs + 1], node offsets per graph
                pretrain_data.groups = torch.cat([
                    torch.full((ptr[i+1] - ptr[i],), i, dtype=torch.long)
                    for i in range(len(ptr) - 1)
                ])

        # Apply GIT's VirtualNodeAugmentor
        vn = VirtualNodeAugmentor()
        pretrain_data, task_node_idx = vn.augment(pretrain_data, task=task)
        pretrain_data = postprocess(pretrain_data)

        train_nodes = task_node_idx
        if params['train_ratio'] != 1:
            train_nodes = torch.tensor(random.sample(train_nodes.tolist(), int(len(train_nodes) * params['train_ratio'])))
        print("Number of training nodes is {}".format(len(train_nodes)))

    encoder = Encoder(
        input_dim=params["input_dim"], hidden_dim=params["hidden_dim"], activation=params["activation"],
        num_layers=params["num_layers"], backbone=params["backbone"], normalize=params["normalize"],
        dropout=params["dropout"]
    )
    feat_decoder = nn.Linear(params["hidden_dim"], params["input_dim"])
    topo_decoder = InnerProductDecoder(hidden_dim=params["hidden_dim"], output_dim=params["hidden_dim"])
    pretrain_model = PretrainModel(encoder=encoder, feat_decoder=feat_decoder, topo_decoder=topo_decoder, ).to(device)

    optimizer = AdamW(pretrain_model.parameters(), lr=params["lr"], weight_decay=params["decay"])
    scheduler = get_scheduler(optimizer, params["use_schedular"], params["epochs"])

    # Resolve checkpoint directory once so intermediate + final saves share it.
    save_path = params.get('save_dir') or osp.join(
        params['model_path'],
        'pretrain_on_{}_seed_{}'.format(params["pretrain_dataset"], params['seed']))
    check_path(save_path)
    save_every = int(params.get('save_every', 0) or 0)
    print("[GIT] Checkpoint dir: {} (save_every={})".format(save_path, save_every))

    # Resume: if --resume and state_latest.pt exists in save_dir, restore full
    # training state and skip already-completed epochs. Note: we restore the
    # full PretrainModel (encoder + sem_encoder + decoders), optimizer, and
    # scheduler, so training continues bit-close to the original run.
    start_epoch = 1
    resume_path = osp.join(save_path, _STATE_LATEST)
    if params.get('resume', False):
        if osp.exists(resume_path):
            last_done = load_resume_state(resume_path, pretrain_model, optimizer, scheduler, device)
            start_epoch = last_done + 1
            print("[GIT] Resumed from {} (last completed epoch={}, continuing at {})".format(
                resume_path, last_done, start_epoch))
            if start_epoch > params["epochs"]:
                print("[GIT] Resume state is already past --epochs={}, nothing to do.".format(params["epochs"]))
        else:
            print("[GIT] --resume set but {} not found, starting fresh.".format(resume_path))

    for i in range(start_epoch, params["epochs"] + 1):
        # In the multi-dataset group path, rebuild the weighted train-node
        # index each epoch so sub-sampled datasets (weights < 1) draw a fresh
        # subset every time. Matches GFT's get_train_node_idx behavior.
        if use_group:
            train_nodes = build_weighted_train_nodes(task_node_idx_dict, weights_dict)
            if i == 1:
                print(
                    "[GIT] Weighted train nodes: total={}, per-dataset={}"
                    .format(
                        len(train_nodes),
                        {k: int(v.numel()) for k, v in task_node_idx_dict.items()},
                    )
                )

        loader = get_loader(pretrain_data, train_nodes, params)

        epoch_stats = pretrain(model=pretrain_model, loader=loader, optimizer=optimizer, scheduler=scheduler, params=params)

        if epoch_stats is not None:
            log_every = int(params.get("log_every", 5))
            if i == 1 or i % log_every == 0 or i == params["epochs"]:
                print(
                    "Epoch {}/{} | loss={:.4f} (feat={:.4f}, topo={:.4f}, sem={:.4f}, align={:.4f})"
                    .format(i, params["epochs"], epoch_stats["total"],
                            epoch_stats["feat"], epoch_stats["topo"],
                            epoch_stats["sem"], epoch_stats["align"])
                )

        # Intermediate checkpoint: every `save_every` epochs + always at final
        # epoch. Two files are written:
        #   - encoder_{i}.pt : encoder-only weights, same naming as GFT / sft.py
        #                      so finetune.py can load via --pt_epochs N.
        #   - state_latest.pt: full training state for --resume. Overwritten
        #                      atomically on every save so only one survives.
        if save_every > 0 and (i % save_every == 0 or i == params["epochs"]):
            inter_path = osp.join(save_path, "encoder_{}.pt".format(i))
            try:
                pretrain_model.save_encoder(inter_path)
                print("  [ckpt] saved {}".format(inter_path))
            except Exception as e:
                print("  [ckpt] WARN failed to save {}: {}".format(inter_path, e))
            try:
                save_resume_state(resume_path, i, pretrain_model, optimizer, scheduler)
                print("  [ckpt] saved {} (epoch={})".format(resume_path, i))
            except Exception as e:
                print("  [ckpt] WARN failed to save {}: {}".format(resume_path, e))

    # Final checkpoint (always written, regardless of save_every).
    pretrain_model.save_encoder(osp.join(save_path, "encoder.pt"))
    print("Saved final model to {}".format(save_path))

    wandb.finish()


if __name__ == "__main__":
    params = get_args_pretrain()
    params['model_path'] = osp.join(osp.dirname(__file__), 'model', 'pretrain_model')

    if params['use_params']:
        config_path = osp.join(osp.dirname(__file__), 'config', 'pretrain.yaml')
        with open(config_path, 'r') as f:
            default_params = yaml.safe_load(f)
            params.update(default_params)

    wandb.init(
        project="GIT-Pretrain",
        name="Pretrain on {} seed={}".format(params["pretrain_dataset"], params["seed"]),
        mode=params.get("wandb_mode", "disabled"),
        config=params,
    )

    run(params)
