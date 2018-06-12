#External Modules
# import os, ase, json, ase.io, collections, math, time
# import numpy as np
# import argparse
import os
import torch
from torch.utils.data import DataLoader
import datasets

#Internal Modules
import CS230_Project.data.database_management as db
from CS230_Project import utils
################################################################################
"""This module contains classes for gathering data for model training"""
################################################################################

'Need all of the necessary environ variables to query the database'




def fetch_dataloader(types, params):
    """
    Fetches the DataLoader object for each type in types from params.data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        params.data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    if not 'dataset' in params.dict.keys():
        params.dataset = 'DFTNetDataset'

    CurrentDataset = getattr(datasets, params.dataset)

    dataloaders = {}

    all_types = ['train', 'val', 'test']

    assert all([t in all_types for t in types])


    for t in types:
        params.data_type = t
        path = os.path.join(params.data_dir, t)

        # use the train_transformer if training data, else use eval_transformer without random flip
        dl = DataLoader(CurrentDataset(params), batch_size=params.batch_size
                , shuffle     = t == 'train' # only shuffle if training
                , num_workers = params.num_workers
                , pin_memory  = params.cuda)

        dataloaders[t] = dl

    return dataloaders
