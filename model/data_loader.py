#External Modules
import os, ase, json, ase.io, collections, math, time
import numpy as np
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

#Internal Modules
import CS230_Project.data.database_management as db
from CS230_Project import utils
################################################################################
"""This module contains classes for gathering data for model training"""
################################################################################

'Need all of the necessary environ variables to query the database'

class DFTNetDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform = None):
        """
        Store the filenames of the sample_data to use. Specifies transforms to apply on inputs.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.torch')]
        self.transform = transform

    def __len__(self): return len(self.filenames) # return size of dataset

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on sample_input.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample_input: (tuple of Tensors) (node_property_tensor, connectivity_tensor, bond_property_tensor)
            label: (Tensor) corresponding formation_energy_per_atom
        """
        filename = self.filenames[idx]
        sample_input, sample_label = torch.load(filename)
        if self.transform:
            sample_input = self.transform(sample_input)
        input_id = filename.split('/')[-1].split('.')[0]
        sample_input_with_id = (sample_input[0], sample_input[1], sample_input[2], sample_input[3], input_id)
        return sample_input_with_id, sample_label


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    all_types = ['train', 'val', 'test']

    assert all([t in all_types for t in types])

    for t in types:
        path = os.path.join(data_dir, t)

        # use the train_transformer if training data, else use eval_transformer without random flip
        dl = DataLoader(DFTNetDataset(path), batch_size=params.batch_size
                , shuffle     = t == 'train' # only shuffle if training
                , num_workers = params.num_workers
                , pin_memory  = params.cuda)

        dataloaders[t] = dl

    return dataloaders
