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

necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
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

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

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
        return sample_input, sample_label


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

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(DFTNetDataset(path), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(DFTNetDataset(path), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders



#ARCHIVE
# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
# parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
# parser.add_argument('--restore_file', default=None,
#                     help="Optional, name of the file in --model_dir containing weights to reload before \
#                     training")  # 'best' or 'train'

# if __name__ == '__main__':
#     args = parser.parse_args()
#     json_path = os.path.join(args.model_dir, 'params.json')
#     assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
#     params = utils.Params(json_path)
#     # use GPU if available
#     params.cuda = torch.cuda.is_available()
#     dataloaders = fetch_dataloader(['train','val'],args.data_dir, params)
