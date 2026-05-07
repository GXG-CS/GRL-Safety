"""GIT inductive KG link-prediction on FB15K237/WN18RR."""
import os.path as osp
import sys

import torch
import torch.nn as nn
import yaml

_GIT_DIR = osp.dirname(osp.abspath(__file__))
_PROJECT_ROOT = osp.abspath(osp.join(_GIT_DIR, "..", "..", ".."))
if _GIT_DIR not in sys.path:
    sys.path.insert(0, _GIT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.args import get_args_finetune
from model.encoder import Encoder
from utils.utils import load_params
from sgb.data.inductive_kg_runner import run_inductive_kg


class _KGModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)


_PARAMS = None


def _builder(in_channels, device):
    act = _PARAMS["activation"]
    if isinstance(act, str):
        act = torch.nn.ReLU if act == "relu" else torch.nn.LeakyReLU
    encoder = Encoder(
        input_dim=_PARAMS["input_dim"], hidden_dim=_PARAMS["hidden_dim"],
        activation=act, num_layers=_PARAMS["num_layers"],
        backbone=_PARAMS["backbone"], normalize=_PARAMS["normalize"],
        dropout=_PARAMS["dropout"],
    )
    ckpt_dir = _PARAMS.get("ckpt_dir")
    if ckpt_dir:
        encoder = load_params(encoder, osp.join(ckpt_dir, "encoder.pt"))
    return _KGModel(encoder).to(device)


def main():
    global _PARAMS
    params = get_args_finetune()
    params['data_path'] = osp.join(osp.dirname(__file__), 'cache_data')
    params['pt_model_path'] = osp.join(osp.dirname(__file__), 'model', 'pretrain_model')
    params['sft_model_path'] = osp.join(osp.dirname(__file__), 'model', 'sft_model')
    params['ft_model_path'] = osp.join(osp.dirname(__file__), 'model', 'finetune_model')
    params['task'] = 'link'

    if params["use_params"]:
        config_path = osp.join(osp.dirname(__file__), "config", f"{params['setting']}.yaml")
        with open(config_path, "r") as f:
            default_params = yaml.safe_load(f)
            params.update(default_params['base'])
            if 'link' in default_params and params['dataset'] in default_params['link']:
                params.update(default_params['link'][params['dataset']])

    _PARAMS = params
    n_seeds = int(params.get("n_seeds", 3))

    run_inductive_kg(
        method_tag="GIT", dataset=params["dataset"],
        build_ft_model=_builder,
        hidden_dim=int(params["hidden_dim"]),
        lr=params.get("lr", 2e-3),
        weight_decay=params.get("weight_decay", 1e-5),
        max_epochs=int(params.get("max_epochs", 600)),
        eval_every=int(params.get("eval_every", 20)),
        patience=int(params.get("patience", 10)),
        n_seeds=n_seeds,
    )


if __name__ == "__main__":
    main()
