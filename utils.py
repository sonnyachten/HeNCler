import collections
from typing import List, Union

import torch_geometric.transforms as T
from sklearn.metrics import cluster
from torch_geometric.data import Data, HeteroData
from torch_geometric.datasets import WikipediaNetwork, HeterophilousGraphDataset, WebKB
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import homophily

from definitions import *


def flatten_dict(d, parent_key='', sep='_', prefix='eval_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep, prefix=prefix).items())
        else:
            items.append((prefix + new_key, v))
    return dict(items)


def pairwise_precision(y_true, y_pred):
    """Computes pairwise precision of two clusterings.

    Args:
    y_true: An [n] int ground-truth cluster vector.
    y_pred: An [n] int predicted cluster vector.

    Returns:
    Precision value computed from the true/false positives and negatives.
    """
    true_positives, false_positives, _, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_positives)


def pairwise_recall(y_true, y_pred):
    """Computes pairwise recall of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      Recall value computed from the true/false positives and negatives.
    """
    true_positives, _, false_negatives, _ = _pairwise_confusion(y_true, y_pred)
    return true_positives / (true_positives + false_negatives)


def pairwise_accuracy(y_true, y_pred):
    """Computes pairwise accuracy of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      Accuracy value computed from the true/false positives and negatives.
    """
    true_pos, false_pos, false_neg, true_neg = _pairwise_confusion(y_true, y_pred)
    return (true_pos + false_pos) / (true_pos + false_pos + false_neg + true_neg)


def _pairwise_confusion(
        y_true,
        y_pred):
    """Computes pairwise confusion matrix of two clusterings.

    Args:
      y_true: An (n,) int ground-truth cluster vector.
      y_pred: An (n,) int predicted cluster vector.

    Returns:
      True positive, false positive, true negative, and false negative values.
    """
    contingency = cluster.contingency_matrix(y_true, y_pred)
    same_class_true = np.max(contingency, 1)
    same_class_pred = np.max(contingency, 0)
    diff_class_true = contingency.sum(axis=1) - same_class_true
    diff_class_pred = contingency.sum(axis=0) - same_class_pred
    total = contingency.sum()

    true_positives = (same_class_true * (same_class_true - 1)).sum()
    false_positives = (diff_class_true * same_class_true * 2).sum()
    false_negatives = (diff_class_pred * same_class_pred * 2).sum()
    true_negatives = total * (
            total - 1) - true_positives - false_positives - false_negatives

    return true_positives, false_positives, false_negatives, true_negatives


def f1_score(y_true, y_pred):
    precision = pairwise_precision(y_true, y_pred)
    recall = pairwise_recall(y_true, y_pred)
    return 2 * precision * recall / (precision + recall)


def load_dataset(name, standardize=False, to_undirected=False, AddRandomWalkPE=False):
    name = name.lower()
    transform = StandardizeFeatures() if standardize else None

    if to_undirected:
        transform = T.Compose([T.ToUndirected()] + [] if transform is None else [transform])

    if AddRandomWalkPE:
        transform = T.Compose(
            [T.AddRandomWalkPE(walk_length=3, attr_name=None)] + [] if transform is None else [transform])

    if name in ['chameleon', 'squirrel']:
        preProcDs = WikipediaNetwork(
            root=DATA_DIR, name=name, geom_gcn_preprocess=False, transform=transform)
        dataset = WikipediaNetwork(
            root=DATA_DIR, name=name, geom_gcn_preprocess=True, transform=transform)
        data = dataset[0]
        data.edge_index = preProcDs[0].edge_index
    elif name in ['texas', 'cornell', 'wisconsin']:
        dataset = WebKB(DATA_DIR, name, transform=transform)
        data = dataset[0]
    elif name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        dataset = HeterophilousGraphDataset(root=DATA_DIR, name=name, transform=transform)
        data = dataset[0]
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')
    return data, dataset.num_classes, dataset.num_features


class StandardizeFeatures(BaseTransform):
    r"""Standardizes the attributes given in :obj:`attrs` column-wise

    Args:
        attrs (List[str]): The names of attributes to normalize.
            (default: :obj:`["x"]`)
    """

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def __call__(
            self,
            data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value = value - value.mean(dim=0)
                value.div_(value.std(dim=0, keepdim=True).clamp_(min=1e-12))
                store[key] = value
        return data


def get_dataset_stats(name_list=None):
    import pandas as pd
    if name_list is None:
        name_list = ['texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel', 'roman-empire', 'amazon-ratings',
                     'minesweeper', 'tolokers', 'questions']
    for i, name in enumerate(name_list):
        data, num_class, num_feats = load_dataset(name)
        stat_dict = {}
        directed = data.is_directed()
        stat_dict.update({
            'num_nodes': data.x.shape[0],
            'num_edges': int(data.edge_index.shape[1] / (1 if directed else 2)),
            'directed': directed,
            'num_feats': data.x.shape[0],
            'num_classes': num_class,
            'homophily_score': homophily(data.edge_index, data.y, method='edge_insensitive')
        })
        if i == 0:
            df = pd.DataFrame(stat_dict, index=[name])
        else:
            df = pd.concat([df, pd.DataFrame(stat_dict, index=[name])])
        df['Heterophilious'] = df['homophily_score'] < 0.2
    return df
