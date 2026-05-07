"""GraphMAE eval only. Loads pretrained model and runs LP eval."""
import logging
import numpy as np
import os
import torch

from graphmae.utils import (
    build_args,
    set_random_seed,
    load_best_configs,
)
from graphmae.datasets.data_util import load_dataset
from graphmae.evaluation import node_classification_evaluation
from graphmae.models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch_f = args.max_epoch_f
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob

    graph, (num_features, num_classes) = load_dataset(dataset_name)
    args.num_features = num_features

    # Load pretrained model
    ckpt_dir = getattr(args, 'ckpt_dir', None) or f"ckpts/graphmae/{dataset_name}"
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    print(f"[GraphMAE eval] Dataset: {dataset_name}, #Features: {num_features}, #Classes: {num_classes}")
    print(f"[GraphMAE eval] Loading model from {ckpt_path}")

    acc_list = []
    estp_acc_list = []
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        model = build_model(args)
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model = model.to(device)
        model.eval()

        x = graph.ndata["feat"]
        final_acc, estp_acc = node_classification_evaluation(
            model, graph, x, num_classes, lr_f, weight_decay_f,
            max_epoch_f, device, linear_prob
        )
        acc_list.append(final_acc)
        estp_acc_list.append(estp_acc)

    final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    print(f"\n=== GraphMAE LP Result ===")
    print(f"# final_acc: {final_acc:.4f} +/- {final_acc_std:.4f}")
    print(f"# early-stopping_acc: {estp_acc:.4f} +/- {estp_acc_std:.4f}")


if __name__ == "__main__":
    args = build_args()
    if args.use_cfg:
        args = load_best_configs(args, "configs_tag.yml")
    print(args)
    main(args)
