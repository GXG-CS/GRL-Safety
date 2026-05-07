import os
import sys
import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

# TAG datasets (from GFT's 8 datasets)
TAG_DATASETS = {'cora', 'pubmed', 'wikics', 'arxiv', 'WN18RR', 'FB15K237', 'chemhiv', 'chempcba'}

# Weights for GFT-style joint pretraining. Mirrors
# sgb/models/gfm/gft/config/pt_data.yaml -> "all" setting. Used by both the
# disjoint-union build step and the per-epoch weighted node sampler.
JOINT_PRETRAIN_WEIGHTS = {
    'all': {
        'cora': 5,
        'pubmed': 5,
        'arxiv': 5,
        'wikics': 5,
        'WN18RR': 5,
        'FB15K237': 10,
        'chemhiv': 1,
        'chemblpre': 0.1,
        'chempcba': 0.1,
    },
}


def get_tag_dataset(name):
    """Load dataset from tag_registry, return (dataset_list, data) compatible with BGRL."""
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag

    data, _ = load_tag(name)

    # Materialize x: index -> SBERT 768d features
    if data.x.dtype == torch.long and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat[data.x]
    elif data.x.ndim == 2 and data.x.size(1) != 768 and hasattr(data, 'node_text_feat'):
        data.x = data.node_text_feat

    # Squeeze y if needed (arxiv has [N,1])
    if hasattr(data, 'y') and data.y is not None and data.y.dim() > 1:
        data.y = data.y.squeeze()

    return [data]


def get_dataset(root, name, transform=NormalizeFeatures()):
    # TAG datasets — defer the whitelist to tag_registry.DATASETS so that any
    # dataset registered there (e.g. products, elephoto, ...) is automatically
    # routable here without having to maintain a second list.
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import DATASETS as _TAG_REGISTRY  # type: ignore

    if name in _TAG_REGISTRY or name in TAG_DATASETS:
        return get_tag_dataset(name)

    # Original PyG datasets
    pyg_dataset_dict = {
        'coauthor-cs': (datasets.Coauthor, 'CS'),
        'coauthor-physics': (datasets.Coauthor, 'physics'),
        'amazon-computers': (datasets.Amazon, 'Computers'),
        'amazon-photos': (datasets.Amazon, 'Photo'),
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name, transform=transform)

    return dataset


def get_wiki_cs(root, transform=NormalizeFeatures()):
    dataset = datasets.WikiCS(root, transform=transform)
    data = dataset[0]
    std, mean = torch.std_mean(data.x, dim=0, unbiased=False)
    data.x = (data.x - mean) / std
    data.edge_index = to_undirected(data.edge_index)
    return [data], np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)


def _normalize_to_index(data):
    """Normalize a single loaded TAG Data so `data.x` is a LongTensor of
    indices into `data.node_text_feat`, and `data.xe` is a LongTensor of
    indices into `data.edge_text_feat`.

    - Node/link datasets (cora, pubmed, arxiv, wikics, WN18RR, FB15K237):
      `data.x` ships as a raw feature matrix (384/128/300/1-d). We discard
      it and use `arange(num_nodes)` since these datasets have one row per
      node in `node_text_feat`.
    - Chem datasets (chemhiv, chemblpre, chempcba): `data.x` is already a
      1-D LongTensor of atom-type indices pointing into `node_text_feat`
      (which is per-atom-type, not per-node). Keep it as is.
    - `xe`: chem ships it; node/link usually dont. If the dataset has
      `edge_types` (KG) use that; otherwise build a zeros tensor of length
      num_edges so every edge maps to the single edge-text-feat row.
    """
    if data.x.dtype != torch.long:
        num_nodes = data.x.shape[0]
        data.x = torch.arange(num_nodes, dtype=torch.long)

    num_edges = data.edge_index.shape[1]
    has_xe = getattr(data, 'xe', None) is not None
    if not has_xe:
        edge_types = getattr(data, 'edge_types', None)
        if edge_types is not None and data.edge_text_feat.shape[0] > 1:
            data.xe = edge_types.long()
        else:
            data.xe = torch.zeros(num_edges, dtype=torch.long)

    return data


def get_joint_pretrain_data(weight_setting='all'):
    """Build a GFT-style disjoint-union of all TAG datasets in the given
    weight setting, for joint SSL pretraining.

    Each dataset's node index space (`x`) and edge-type index space (`xe`)
    are shifted by the cumulative size of the preceding datasets
    `node_text_feat` / `edge_text_feat` tables, then the whole list is
    merged via `Batch.from_data_list`. PyG handles offsetting `edge_index`
    automatically. `node_text_feat` and `edge_text_feat` are concatenated
    into single global tables so that a batch sampled by NeighborLoader can
    materialize features lazily via `global_node_text_feat[batch.x]`.

    Importantly, `x` stays a LongTensor throughout — we never eagerly
    materialize the full per-node feature table, which avoids the 27GB
    allocation chemblpre would otherwise need.

    Returns:
        big_data: PyG Data with
            - x: global LongTensor indices into node_text_feat
            - edge_index: offset per PyG Batch rules
            - xe: global LongTensor indices into edge_text_feat
            - ptr: per-dataset node boundaries (from Batch.from_data_list)
        node_text_feat: (total_unique_nodes, 768) concatenated table (kept
            outside big_data so NeighborLoader doesnt try to slice it)
        edge_text_feat: (total_edge_types, 768) concatenated table
        dataset_names: ordered list of datasets included
        weights: dict of dataset_name -> sampling weight (for the sampler)
    """
    from torch_geometric.data import Batch

    if weight_setting not in JOINT_PRETRAIN_WEIGHTS:
        raise ValueError(
            f"Unknown weight_setting '{weight_setting}'. "
            f"Available: {list(JOINT_PRETRAIN_WEIGHTS.keys())}"
        )
    weights = JOINT_PRETRAIN_WEIGHTS[weight_setting]
    dataset_names = list(weights.keys())

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    from sgb.data.tag_registry import load as load_tag

    data_list = []
    for name in dataset_names:
        data, _ = load_tag(name)
        data = _normalize_to_index(data)
        data_list.append(data)

    # Offset x and xe into the global feature-table index space.
    x_start, xe_start = 0, 0
    for data in data_list:
        num_unique_nodes = data.node_text_feat.shape[0]
        num_edge_types = data.edge_text_feat.shape[0]
        data.x = data.x + x_start
        data.xe = data.xe + xe_start
        x_start += num_unique_nodes
        xe_start += num_edge_types

    # Concatenate the global feature tables before stripping them off each
    # per-dataset Data, so Batch.from_data_list doesnt try to collate them.
    global_node_text_feat = torch.cat([d.node_text_feat for d in data_list], dim=0)
    global_edge_text_feat = torch.cat([d.edge_text_feat for d in data_list], dim=0)
    for d in data_list:
        d.node_text_feat = None
        d.edge_text_feat = None

    # Keep only the minimal fields needed for joint pretraining to avoid
    # Batch.from_data_list choking on dataset-specific extras (raw_texts,
    # train_mask of varying shapes, y of mismatched dtypes, etc.).
    keep_keys = {'x', 'edge_index', 'xe'}
    clean_list = []
    for d in data_list:
        kept = {k: v for k, v in d.to_dict().items() if k in keep_keys and v is not None}
        clean_list.append(type(d)(**kept))

    big_data = Batch.from_data_list(clean_list)

    return big_data, global_node_text_feat, global_edge_text_feat, dataset_names, weights


def get_joint_train_nodes(big_data, dataset_names, weights):
    """GFT-style per-dataset weighted node sampler.

    For each dataset, repeat its node-index block `int(weight)` times and
    add an extra `frac(weight) * N` random subsample. The concatenation of
    all per-dataset indices is the `input_nodes` list for NeighborLoader.

    The dataset boundaries come from `big_data.ptr` (which
    Batch.from_data_list populated when we merged the disjoint union).

    Called once per epoch to re-randomize the fractional subsamples.
    """
    assert big_data.ptr is not None, "big_data.ptr missing — was it built via Batch.from_data_list?"
    assert len(big_data.ptr) - 1 == len(dataset_names), (
        f"ptr has {len(big_data.ptr)-1} segments but {len(dataset_names)} datasets"
    )

    pieces = []
    for idx, name in enumerate(dataset_names):
        w = weights[name]
        s, e = int(big_data.ptr[idx].item()), int(big_data.ptr[idx + 1].item())
        arr = torch.arange(s, e)
        int_w, frac_w = int(w), w - int(w)

        repeated = arr.repeat(int_w) if int_w > 0 else arr.new_empty(0)
        if frac_w > 0:
            n_frac = int(frac_w * arr.size(0))
            if n_frac > 0:
                frac_idx = arr[torch.randperm(arr.size(0))[:n_frac]]
            else:
                frac_idx = arr.new_empty(0)
        else:
            frac_idx = arr.new_empty(0)
        pieces.append(torch.cat([repeated, frac_idx]))

    return torch.cat(pieces)


class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)
