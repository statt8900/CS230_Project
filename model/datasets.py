from torch.utils.data import Dataset
import torch
import os
import cPickle as pkl
from torch.autograd import Variable
import numpy as np

class OqmdFormationEnergyDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params):
        """
        Store the filenames of the sample_data to use. Specifies transforms to apply on inputs.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        super(OQMDDataset, self).__init__()
        self.required_keys = ['data_dir', 'data_type']
        for key in self.required_keys:
            assert key in params.dict.keys()

        self.ids = torch.load(os.path.join(params.data_dir, params.data_type, 'all_structure_ids.torch'))


    def __len__(self):
        return len(self.ids) # return size of dataset

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on sample_input.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample_input: (tuple of Tensors) (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_id)
            label: (Tensor) corresponding formation_energy_per_atom
        """
        structure_id = self.ids[idx]
        node_property_tensor = torch.load(os.path.join(params.data_dir, params.data_type, str(structure_id) + '_node_property_tensor.torch'))
        connectivity_tensor = torch.load(os.path.join(params.data_dir, params.data_type, str(structure_id) + '_connectivity_tensor.torch'))
        bond_property_tensor = torch.load(os.path.join(params.data_dir, params.data_type, str(structure_id) + '_bond_property_tensor.torch'))
        mask_atom_tensor = torch.load(os.path.join(params.data_dir, params.data_type, str(structure_id) + '_mask_atom_tensor.torch'))
        e_form = torch.load(os.path.join(params.data_dir, params.data_type, str(structure_id) + '_e_form.torch'))
        sample_input_with_id = (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, structure_id)
        return sample_input_with_id, e_form

class DFTNetDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params, transform = None):
        """
        Store the filenames of the sample_data to use. Specifies transforms to apply on inputs.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.required_keys = ['data_dir', 'data_type']
        for key in self.required_keys:
            assert key in params.dict.keys()

        data_dir = os.path.join(params.data_dir, params.data_type)
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
            sample_input: (tuple of Tensors) (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_id)
            label: (Tensor) corresponding formation_energy_per_atom
        """
        filename = self.filenames[idx]
        sample_input, sample_label = torch.load(filename)
        if self.transform:
            sample_input = self.transform(sample_input)
        input_id = filename.split('/')[-1].split('.')[0]
        sample_input_with_id = (sample_input[0], sample_input[1], sample_input[2], sample_input[3], input_id)
        return sample_input_with_id, sample_label



class ErrorNetDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params):
        """
        Store the filenames of the sample_data to use. Specifies transforms to apply on inputs.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        stats = torch.load(os.path.join(params.model_dir, 'dataset_statistics.torch'))
        self.means = stats[0,:].type(torch.FloatTensor)
        self.stdevs = stats[1,:].type(torch.FloatTensor)

        self.params = params
        self.required_keys = ['data_dir', 'data_type', 'layer']
        for key in self.required_keys:
            assert key in params.dict.keys()

        if 'cuda' in params.dict.keys():
            self.cuda = params.cuda
        else:
            self.cuda = torch.cuda.is_available()

        self.all_filenames = os.listdir(os.path.join(params.data_dir, params.data_type, params.layer))
        self.filenames = [os.path.join(params.data_dir, params.data_type, params.layer, f) for f in self.all_filenames if f.endswith('.torch')]
        if 'transform' in params.dict.keys():
            self.transform = params.transform
        else:
            self.transform = None

        self.error_dict = pkl.load(open(os.path.join(params.data_dir, '../error_analysis', params.data_type, 'error_dict.pkl'),'rb'))
        self.n_atoms_vector = np.load(os.path.join(params.data_dir, '../error_analysis', params.data_type, 'n_atoms.npy'))

    def __len__(self): return len(self.filenames) # return size of dataset

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on sample_input.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample_input: (tuple of Tensors) (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_id)
            label: (Tensor) corresponding formation_energy_per_atom
        """
        filename = self.filenames[idx]
        activation_tensor = torch.load(filename)
        activation_tensor -= self.means
        activation_tensor /= self.stdevs

        (nrows, ncols) = activation_tensor.shape

        ID = filename.split('/')[-1].split('.')[0]
        absolute_error = self.error_dict[ID][0]*torch.ones(1)
        actual_energy = self.error_dict[ID][1]*torch.ones(1)
        predicted_energy = self.error_dict[ID][2]*torch.ones(1)

        n_atoms = int(self.error_dict[ID][3])
        mask_atom_tensor = torch.zeros(nrows)
        mask_atom_tensor[0:n_atoms] = torch.ones(n_atoms)

        if self.cuda:
            raise NotImplementedError
            # return ((activation_tensor.cuda(async=True), mask_atom_tensor.cuda(async=True)), (predicted_energy.cuda(async=True), actual_energy.cuda(async=True)))
        else:
            # return ((activation_tensor, mask_atom_tensor), (predicted_energy, actual_energy))
            return ((activation_tensor, mask_atom_tensor), absolute_error)

class ErrorNetConvDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, params):
        """
        Store the filenames of the sample_data to use. Specifies transforms to apply on inputs.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        stats = torch.load(os.path.join(params.model_dir, 'dataset_statistics.torch'))
        self.means = stats[0,:].type(torch.FloatTensor)
        self.stdevs = stats[1,:].type(torch.FloatTensor)

        self.params = params
        self.required_keys = ['data_dir', 'data_type']
        for key in self.required_keys:
            assert key in params.dict.keys()

        if 'cuda' in params.dict.keys():
            self.cuda = params.cuda
        else:
            self.cuda = torch.cuda.is_available()

        self.all_filenames = os.listdir(os.path.join(params.data_dir, params.data_type))
        self.filenames = [os.path.join(params.data_dir, params.data_type, f) for f in self.all_filenames if f.endswith('.torch')]
        if 'transform' in params.dict.keys():
            self.transform = params.transform
        else:
            self.transform = None

        self.error_dict = pkl.load(open(os.path.join(params.model_dir, '../error_analysis', params.data_type, 'error_dict.pkl'),'rb'))
        # self.n_atoms_vector = np.load(os.path.join(params.model_dir, '../error_analysis', params.data_type, 'n_atoms.npy'))

        ### Exclude outliers
        # for key, tup in self.error_dict.items():
        #     if tup[0] > .2:
        #         self.filenames.remove(os.path.join(params.data_dir, params.data_type, key + '.torch'))

    def __len__(self): return len(self.filenames) # return size of dataset

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on sample_input.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            sample_input: (tuple of Tensors) (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor, input_id)
            label: (Tensor) corresponding formation_energy_per_atom
        """
        filename = self.filenames[idx]
        sample_input, sample_label = torch.load(filename)

        ID = filename.split('/')[-1].split('.')[0]
        absolute_error = self.error_dict[ID][0]*torch.ones(1)
        actual_energy = self.error_dict[ID][1]*torch.ones(1)
        predicted_energy = self.error_dict[ID][2]*torch.ones(1)

        sample_input_with_id = (sample_input[0], sample_input[1], sample_input[2], sample_input[3], ID)


        if self.cuda:
            raise NotImplementedError
            # return ((activation_tensor.cuda(async=True), mask_atom_tensor.cuda(async=True)), (predicted_energy.cuda(async=True), actual_energy.cuda(async=True)))
        else:
            # return ((activation_tensor, mask_atom_tensor), (predicted_energy, actual_energy))
            return (sample_input_with_id, absolute_error)
