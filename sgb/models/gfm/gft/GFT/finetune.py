#!/usr/bin/env python
# coding: utf-8

import os
import os.path as osp
import yaml
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from dataset.process_datasets import get_finetune_graph
from model.encoder import Encoder
from model.vq import VectorQuantize
from model.ft_model import TaskModel
from utils.loader import get_loader
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.args import get_args_finetune
from utils.preprocess import pre_node, pre_link, pre_graph
from utils.others import seed_everything, load_params, mask2idx
from utils.splitter import get_split, get_split_graph

from task.node import ft_node, eval_node
from task.link import ft_link, eval_link
from task.graph import ft_graph, eval_graph

import warnings
import wandb

warnings.filterwarnings("ignore")

dataset2task = {
    "cora": "node",
    "pubmed": "node",
    "arxiv": "node",
    "wikics": "node",
    "WN18RR": "link",
    "FB15K237": "link",
    "chemhiv": "graph",
    "chempcba": "graph",
}


def get_preprocess(params):
    if params['task'] == 'node':
        return pre_node
    elif params['task'] == 'link':
        return pre_link
    elif params['task'] == 'graph':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')


def get_ft(params):
    task = params['task']

    if task == "node":
        return ft_node
    elif task == "link":
        return ft_link
    elif task == "graph":
        return ft_graph
    else:
        raise ValueError("Invalid Task")


def get_eval(params):
    task = params['task']

    if task == "node":
        return eval_node
    elif task == "link":
        return eval_link
    elif task == "graph":
        return eval_graph
    else:
        raise ValueError("Invalid Task")


def run(params):
    params["activation"] = nn.ReLU if params["activation"] == "relu" else nn.LeakyReLU
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['activation'] = nn.ReLU if params['activation'] == 'relu' else nn.LeakyReLU

    preprocess = get_preprocess(params)
    finetune = get_ft(params)
    evaluate = get_eval(params)

    data_name = params["finetune_dataset"]
    task = params["task"]
    setting = params["setting"]

    dataset, splits, labels, num_classes, num_tasks = get_finetune_graph(params['data_path'], data_name)
    num_classes = num_tasks if task == "graph" else num_classes
    params["num_classes"] = num_classes

    dataset = preprocess(dataset)
    data = dataset[0]
    data.y = labels

    if isinstance(splits, list):
        pass
    elif isinstance(splits, dict):
        splits = [splits] * params["repeat"]

    encoder = Encoder(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        activation=params["activation"],
        num_layers=params["num_layers"],
        backbone=params["backbone"],
        normalize=params["normalize"],
        dropout=params["dropout"],
    )

    vq = VectorQuantize(
        dim=params["hidden_dim"],
        codebook_size=params["codebook_size"],
        codebook_dim=params["code_dim"],
        heads=params["codebook_head"],
        separate_codebook_per_head=True,
        decay=params["codebook_decay"],
        commitment_weight=params["commit_weight"],
        use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
        orthogonal_reg_weight=params["ortho_reg_weight"],
        orthogonal_reg_max_codes=params["ortho_reg_max_codes"],
        orthogonal_reg_active_codes_only=False,
        kmeans_init=True,
        ema_update=False,
    )

    # Load Pretrained Model
    # GFM-Safety: flat ckpt layout at <repo>/ckpts/GFT/{encoder,vq}.pt rather
    # than the original GFT's nested `codebook_size_*_layer_*_pretrain_on_*`
    # scheme. pretrain_dataset / pretrain_model_epoch / pretrain_seed args are
    # kept for CLI back-compat but no longer affect the loaded file name.
    if params["pretrain_dataset"] != 'na':
        path = osp.abspath(osp.join(
            osp.dirname(__file__), "..", "..", "..", "..", "ckpts", "GFT"
        ))
        encoder = load_params(encoder, osp.join(path, 'encoder.pt'))
        vq = load_params(vq, osp.join(path, 'vq.pt'))
        print("Loaded pretrained encoder and vq from {}".format(path))

    train_loader = None
    val_loader = None
    test_loader = None
    subgraph_loader = None

    if params["batch_size"] == 0:
        data = data.to(device)
        labels = labels.to(device)

    logger = Logger()

    for idx, split in enumerate(splits):
        seed_everything(idx)

        if setting == "standard":
            split = split
        elif setting in ["few_shot", "zero_shot", "in_context"]:
            if task in ["node", "link"]:
                split = get_split(split, labels, params)
            elif task == "graph":
                split = get_split_graph(split, labels, params)
        else:
            raise ValueError("Invalid Setting")

        task_model = TaskModel(
            encoder=deepcopy(encoder),
            vq=deepcopy(vq),
            num_classes=num_classes,
            params=params,
        ).to(device)

        opt_params = task_model.parameters()
        task_opt = AdamW(opt_params, lr=params["finetune_lr"])
        stopper = EarlyStopping(patience=params["early_stop"])

        if params["batch_size"] != 0 and task in ["node", "link"]:
            train_loader, subgraph_loader = get_loader(data, split, labels, params)
        elif params["batch_size"] != 0 and task == "graph":
            train_loader, val_loader, test_loader = get_loader(dataset, split, labels, params)

        for epoch in range(params["finetune_epochs"]):
            loss = finetune(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=train_loader,
                optimizer=task_opt,
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[30] * params["num_layers"],
            )

            result = evaluate(
                model=task_model,
                dataset=data if task in ["node", "link"] else dataset,
                loader=subgraph_loader if task in ["node", "link"] else [train_loader, val_loader, test_loader],
                split=split,
                labels=labels,
                params=params,
                num_neighbors=[-1] * params["num_layers"],
            )

            is_stop = stopper(result)
            logger.log(idx, epoch, loss, result)
            if is_stop:
                print("Early Stopping at Epoch:", epoch)
                break

            wandb.log(
                {
                    "train/proto_loss": loss['proto_loss'],
                    "train/lin_loss": loss['act_loss'],
                    "train/loss": loss['loss'],
                    "train/train_value": result['train'],
                    "train/val_value": result['val'],
                    "train/test_value": result['test'],
                }
            )

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best/train": single_best["train"],
            "best/val": single_best["val"],
            "best/test": single_best["test"],
        })

    best = logger.get_best()

    # GFM-Safety: also print final result to stdout so --debug runs surface the numbers.
    print("\n=== Final (aggregated over {} repeats) ===".format(params["repeat"]))
    print("  train: {:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']))
    print("  val:   {:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']))
    print("  test:  {:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']))

    wandb.log({
        "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final/train_mean": best['train']['mean'],
        "final/val_mean": best['val']['mean'],
        "final/test_mean": best['test']['mean'],
        "final/train_std": best['train']['std'],
        "final/val_std": best['val']['std'],
        "final/test_std": best['test']['std'],
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})

    wandb.finish()


if __name__ == "__main__":
    params = get_args_finetune()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pt_model_path'] = osp.join(osp.dirname(__file__), '..', 'ckpts', 'pretrain_model')

    dataset = params["finetune_dataset"]
    task = dataset2task[dataset]
    params['task'] = task

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'finetune.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[task][dataset])

    if params["setting"] in ["few_shot"]:
        if params['finetune_dataset'] in ['FB15K237']:
            params['batch_size'] = 0
        if task == 'graph':
            params['n_way'] = 2
            params['num_instances_per_class'] = params['n_train']

    # At least use a classifier
    assert not (params['no_lin_clf'] and params['no_proto_clf'])
    if params['no_lin_clf']:
        params['trade_off'] = 0
    if params['no_proto_clf']:
        params['trade_off'] = 1

    wandb.init(
        project="GFT-Finetune",
        name="{} - Pretrain Epoch {}".format(str.upper(params["finetune_dataset"]), params["pretrain_model_epoch"]),
        config=params,
        mode="disabled" if params["debug"] else "online",  # sweep only works in online mode
        tags=[params['setting']],
    )
    params = dict(wandb.config)
    print(params)

    run(params)
