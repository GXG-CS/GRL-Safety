"""GraphMAE pretrain only. Saves model to ckpt_dir/model.pt"""
import logging
import numpy as np
import os
from tqdm import tqdm
import torch

from graphmae.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    TBLogger,
    get_current_lr,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device, scheduler, logger=None):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()

        loss, loss_dict = model(graph, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate

    optim_type = args.optimizer
    loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    linear_prob = args.linear_prob
    load_model = args.load_model
    save_model = args.save_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    print(f"[GraphMAE pretrain] Dataset: {dataset_name}, #Features: {num_features}, #Classes: {num_classes}")
    print(f"[GraphMAE pretrain] Encoder: {encoder_type}, Hidden: {num_hidden}, Layers: {num_layers}")

    # Use first seed for pretrain
    seed = seeds[0] if seeds else 0
    print(f"[GraphMAE pretrain] Seed: {seed}")
    set_random_seed(seed)

    if logs:
        logger = TBLogger(name=f"{dataset_name}_pretrain_{encoder_type}_{num_hidden}_{num_layers}")
    else:
        logger = None

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / max_epoch)) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    x = graph.ndata["feat"]
    if not load_model:
        model = pretrain(model, graph, x, optimizer, max_epoch, device, scheduler, logger)
        model = model.cpu()

    if load_model:
        logging.info("Loading Model ...")
        model.load_state_dict(torch.load("checkpoint.pt"))

    # Save model
    ckpt_dir = getattr(args, 'ckpt_dir', None) or f"ckpts/graphmae/{dataset_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[GraphMAE pretrain] Saved model to {ckpt_path}")

    if logger is not None:
        logger.finish()


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs_tag.yml")
    print(args)
    main(args)
