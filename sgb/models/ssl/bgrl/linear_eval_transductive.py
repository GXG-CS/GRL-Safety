import logging

from absl import app
from absl import flags
import numpy as np
import torch

from bgrl import *

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics', 'wiki-cs',
                   'cora', 'pubmed', 'wikics', 'arxiv'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', None, 'Conv layer sizes.')
flags.DEFINE_string('ckpt_path', None, 'Path to checkpoint.')

# TAG datasets
TAG_NODE_DATASETS = {'cora', 'pubmed', 'wikics', 'arxiv'}


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for evaluation.'.format(device))

    # load data
    if FLAGS.dataset in TAG_NODE_DATASETS:
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    elif FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(FLAGS.dataset, data))
    data = data.to(device)  # permanently move in gpu memory

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True)

    # load pretrained encoder
    ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        encoder.load_state_dict(ckpt['model'])
    else:
        encoder.load_state_dict(ckpt)
    encoder = encoder.to(device)
    encoder.eval()
    print(f"[BGRL eval] Loaded encoder from {FLAGS.ckpt_path}")
    print(f"[BGRL eval] Dataset: {FLAGS.dataset}, input_dim={input_size}, repr_dim={representation_size}")

    # compute representations
    representations, labels = compute_representations(encoder, dataset, device)

    if FLAGS.dataset in TAG_NODE_DATASETS:
        # Use TAG data's native splits
        data = dataset[0]
        if hasattr(data, 'train_masks'):
            # cora/pubmed: list of 10 splits -> convert to numpy [N, n_splits]
            import torch as _torch
            tm = _torch.stack(data.train_masks, dim=1).cpu().numpy()
            vm = _torch.stack(data.val_masks, dim=1).cpu().numpy()
            tsm = _torch.stack(data.test_masks, dim=1).cpu().numpy()[:, 0]  # test shared
            scores = fit_logistic_regression_preset_splits(
                representations.cpu().numpy(), labels.cpu().numpy(), tm, vm, tsm)
        elif hasattr(data, 'train_mask'):
            tm = data.train_mask.cpu()
            vm = data.val_mask.cpu()
            tsm = data.test_mask.cpu()
            # Single split or WikiCS [N,20] multi-split
            if tm.dim() == 2:
                scores = fit_logistic_regression_preset_splits(
                    representations.cpu().numpy(), labels.cpu().numpy(),
                    tm.numpy(), vm.numpy(), tsm.numpy())
            else:
                # Single split (arxiv): repeat 5 times with same split
                scores = []
                X = normalize(representations.cpu().numpy(), norm='l2')
                y_np = labels.cpu().numpy()
                from sklearn.linear_model import LogisticRegression
                from sklearn.multiclass import OneVsRestClassifier
                from sklearn.preprocessing import OneHotEncoder as OHE
                ohe = OHE(categories='auto', sparse_output=False)
                y_oh = ohe.fit_transform(y_np.reshape(-1, 1)).astype(np.bool_)
                X_train, y_train = X[tm], y_oh[tm]
                X_val, y_val = X[vm], y_oh[vm]
                X_test, y_test = X[tsm], y_oh[tsm]
                best_test_acc = 0
                for c in 2.0 ** np.arange(-10, 11):
                    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
                    clf.fit(X_train, y_train)
                    y_pred = np.argmax(clf.predict_proba(X_val), axis=1)
                    y_pred = ohe.transform(y_pred.reshape(-1, 1)).astype(np.bool_)
                    val_acc = (y_pred == y_val).all(axis=1).mean()
                    if val_acc > best_test_acc:
                        best_test_acc = val_acc
                        y_p = np.argmax(clf.predict_proba(X_test), axis=1)
                        y_p = ohe.transform(y_p.reshape(-1, 1)).astype(np.bool_)
                        final_acc = (y_p == y_test).all(axis=1).mean()
                scores = [final_acc]
        else:
            scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                             data_random_seed=1, repeat=5)
    elif FLAGS.dataset == 'wiki-cs':
        scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                       train_masks, val_masks, test_masks)
    else:
        scores = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                         data_random_seed=1, repeat=5)

    print(f"\n=== BGRL LP Result ===")
    print(f"Test: {np.mean(scores)*100:.2f} +/- {np.std(scores)*100:.2f}")


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
