
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, add_self_loops, homophily
import numpy as np
from torch_geometric.datasets import (
    WikipediaNetwork,
    CitationFull,
    Amazon
)
from datahelper.datasets import Squirrel
import torch
import os.path as osp

def get_dataset(name, root_dir, self_loops, undirected):
    path = f"{root_dir}/"
    if name in ["chameleon"]:
        dataset = WikipediaNetwork(root = path, name = name, transform = T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['squirrel']:
        dataset = Squirrel(root = path, name = name, transform = T.NormalizeFeatures())
        data = dataset[0]        
    elif name == 'photo':
        dataset = Amazon(root = path, name = name, transform = T.NormalizeFeatures())
        data = dataset[0]
    elif name in ['citeseer', 'cora_ml']:
        dataset = CitationFull(root = path, name = name, transform = T.NormalizeFeatures())
        data = dataset[0]
    homo = homophily(data.edge_index, data.y)
    print('homophily ratio: ' + str(homo))
    if undirected:
        data.edge_index = to_undirected(data.edge_index)
    if self_loops:
        data.edge_index = add_self_loops(data.edge_index)

    return dataset, data


def get_split(name, data, mask_id):
    if name in ["chameleon", "squirrel"]:
        return data, data.train_mask[:, mask_id], data.val_mask[:, mask_id], data.test_mask[:, mask_id]
    # if name in [ "squirrel"]:       
    #     tmp_val_mask = data.val_mask[:, mask_id]
    #     tmp_val_mask[:3500] = 0
    #     return data, data.train_mask[:, mask_id] + tmp_val_mask, data.val_mask[:, mask_id], data.test_mask[:, mask_id]
    elif name in ['citeseer', 'cora_ml' , 'photo']:
        labels = data.y.numpy()
        mask = train_test_split(labels=labels, seed=1020, train_examples_per_class=20, val_size=500, test_size=None)

        mask['train'] = torch.from_numpy(mask['train']).bool()
        mask['val'] = torch.from_numpy(mask['val']).bool()
        mask['test'] = torch.from_numpy(mask['test']).bool()

        data.train_mask = mask['train']
        data.val_mask = mask['val']
        data.test_mask = mask['test']
        return data, data.train_mask, data.val_mask, data.test_mask



def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None, test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size, val_size, test_size)

    #print('number of training: {}'.format(len(train_indices)))
    #print('number of validation: {}'.format(len(val_indices)))
    #print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices

def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max()+1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])
