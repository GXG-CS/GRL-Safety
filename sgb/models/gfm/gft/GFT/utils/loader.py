from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
from utils.others import seed_everything, load_params, mask2idx


def get_loader(data, split, labels, params):
    task = params['task']
    setting = params["setting"]

    if task == "node":
        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10] * params["num_layers"],
                input_nodes=mask2idx(split["train"]),
                batch_size=params["batch_size"],
                num_workers=0,
                shuffle=True,
            )
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            batch_size=512,
            num_workers=0,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "link":
        # If the dataset has only one edge type (binary link prediction such as
        # protein_hs/ml1m), all `edge_label` rows are 1, which makes AUC/F1
        # degenerate. Inject 1:1 negative sampling for both train and eval.
        single_relation = (labels.unique().numel() <= 1)
        neg_kwargs = {"neg_sampling_ratio": 1.0} if single_relation else {}

        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = LinkNeighborLoader(
                data,
                num_neighbors=[30] * params["num_layers"],
                edge_label_index=data.edge_index[:, split["train"]],
                edge_label=labels[split["train"]],
                batch_size=params["batch_size"],
                num_workers=0,
                shuffle=True,
                **neg_kwargs,
            )
        subgraph_loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * params["num_layers"],
            edge_label_index=data.edge_index,
            edge_label=labels,
            batch_size=4096,
            num_workers=0,
            shuffle=False,
            **neg_kwargs,
        )
        return train_loader, subgraph_loader

    elif task == "graph":
        if setting == 'standard':
            train_dataset = data[split["train"]]
            val_dataset = data[split["valid"]]
            test_dataset = data[split["test"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=params["batch_size"],
                shuffle=False,
                num_workers=0,
            )
        elif setting in ['few_shot']:
            # As we only update the train_idx in sampling few-shot samples,
            # we can directly use the split["train"] as the train_idx
            # This enables the shuffle function in DataLoader.
            # The drawback is we should define the proto_loader in the finetune_graph_task function
            train_dataset = data[split["train"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=params["batch_size"],
                shuffle=True,
                num_workers=0,
            )
            val_loader = None
            test_loader = None

        elif setting in ['zero_shot', 'in_context']:
            train_loader = None
            val_loader = None
            test_loader = None

        return train_loader, val_loader, test_loader
