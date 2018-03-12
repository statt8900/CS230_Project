"""Split the SIGNS dataset into train/val/test and resize images to 64x64.

The SIGNS dataset comes into the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and val sets.
Because we don't have a lot of images and we want that the statistics on the val set be as
representative as possible, we'll take 20% of "train_signs" as val set.
"""
#External Modules
import os, json, shutil
import torch
import argparse
import random
import os
from sklearn.model_selection import train_test_split
from mendeleev import element
from tqdm import tqdm

#Internal Imports
from CS230_Project.misc.sql_shortcuts import *
import CS230_Project.data.database_management as db
from CS230_Project.misc.utils import traj_rebuild

project_folder  = os.environ['CS230_Project_Folder']
datasets_folder = os.environp['CS230_Datasets']

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default=os.path.join(datasets_folder,'/dataset'), help="Where to write the new data")
parser.add_argument('--data_dir', default=os.path.join(datasets_folder,'/raw_inputs'), help="Where to read the raw data")

#Sorting Functions for determining bond importance
bo_over_dis = lambda distance, bondorder: bondorder/distance

class RawDataExtractor(object):
    """
    Subclass of Torch Dataset object
    storage_directories :: list of storage directories where chargemol analysis,
                            result.json, and final.traj are stored
    """
    def __init__(self,constraints = [], limit = 10, filter_length = 13, num_atoms_after_padding = 100):
        default_constraints = [PMG_Entries.chargemol==1, PMG_Entries.num_atoms<=num_atoms_after_padding]
        default_constraints += constraints
        self.query          = db.Query(constraints = default_constraints, limit = limit, verbose = True)
        self.output_dict    = self.query.query_dict(cols = ['*'])
        self.material_ids   = self.query.query_col(PMG_Entries.material_id)
        self.num_atoms_after_padding      = num_atoms_after_padding
        self.filter_length  = filter_length
        self.attributes     = ['en_pauling'
                              ,'dipole_polarizability'
                              ,'melting_point'
                              ,'boiling_point'
                              ,'covalent_radius'
                              ,'period'
                              ,'group_id']

    def __getitem__(self, material_id):
        return self.row_dict_to_CNN_input(self.output_dict[self.material_ids.index(material_id)])

    def __len__(self):
        return len(self.output_dict)

    def row_dict_to_CNN_input(self, row_dict):
        """
        Take a storage directory, with chargemol_analysis and job_output subfolders,
        and produce a connectivity matrix
        """
        material_id          = row_dict['material_id']

        #Create tensor to mask the dummy atoms added for padding
        mask_atom_tensor     = torch.zeros(self.num_atoms_after_padding)
        mask_atom_tensor[:row_dict['num_atoms']] = 1

        #Store the formation_energy_per_atom
        e_form               = row_dict['formation_energy_per_atom']*row_dict['num_atoms']/self.num_atoms_after_padding*torch.ones(1)

        #Extract the connectivity
        connectivity_tensor, bond_property_tensor = self.row_dict_to_connectivity_tensors(row_dict)

        #Get the node feature matrix
        atoms_obj            = traj_rebuild(row_dict['atoms_obj'])
        node_property_tensor = self.atoms_to_node_properties(atoms_obj)

        return (node_property_tensor, connectivity_tensor, bond_property_tensor, mask_atom_tensor), e_form

    def atoms_to_node_properties(self, atoms):
        """
        Converts atoms object into a numpy array
        node_property_tensor is shape (num_atoms_after_padding,number_of_properties)
        """

        node_property_tensor        = torch.zeros(self.num_atoms_after_padding, len(self.attributes))
        for (i,atom) in enumerate(atoms):
            node_property_tensor[i] = self.get_atom_properties(atom)
        return node_property_tensor


    def get_atom_properties(self, atom):
        """
        returns an PyTorch tensor of length len(self.attributes)
        the features are pulled from mendeleev element object
        (See the attributes member data in __init__ for full list of attributes)
        """
        element_obj             = element(atom.symbol)
        properties              = torch.zeros(len(self.attributes))

        for i, attr in enumerate(self.attributes):
            if element_obj.__getattribute__(attr) == None:
                print atom.symbol
                print attr
            else:
                properties[i]   = element_obj.__getattribute__(attr)
        return properties

    @staticmethod
    def _get_sorted_bond_tensor(bond_dict, bond_function = bo_over_dis):
        """
        Convert bond dictionary into a pytorch tensor sorted by a function of
        distance and bondorder
        input
        bond_dict          :: dict with keys: fromNode, toNode, distance, bondorder
        bond_function      :: function that takes distance, bondorder as inputs (in that order)
                              and outputs value that is higher for more important bonds

        output:
        sorted_bond_tensor :: pytorch tensor sorted by the bond function output (highest to lowest)

        """
        #convert dict into tensor
        unpacked_bond_dict  = [[bond['fromNode'],bond['toNode'],bond['distance'],bond['bondorder']] for bond in bond_dict]
        bond_tensor         = torch.Tensor(unpacked_bond_dict)

        #Apply bond function and sort the bond_tensor by that value
        bond_sort_value     = torch.Tensor(map(bond_function,*zip(*bond_tensor[:,2:])))
        _, sort_indices     = torch.sort(bond_sort_value, descending = True)
        sorted_bond_tensor  = bond_tensor[sort_indices,:]
        return sorted_bond_tensor

    def row_dict_to_connectivity_tensors(self, row_dict,bond_order_cutoff = 0.01):
        """
        Uses bonds.json to produce two tensors that contain all of the connectivity
        information.
        input:
        row_dict                :: dict from the database
        bond_order_cutoff       :: float, bonds below this cutoff will not be
                                   added to connectivity tensors

        output:
        connectivity_tensor     :: Type     = pytorch tensor
                                :: Shape    = (n_atoms, n_atoms, filter_length)
        Description: Each atom gets a hot_one tensor that can be dotted with the
        node_property_tensor to create a matrix that can be convulved with a filter

        bond_property_tensor    :: Type     = pytorch tensor
                                :: Shape    = (n_atoms, filter_length, 2)
        Description: Each atom gets a filter_length by 2 tensor that has bond properties
        (distance, bondorder) listed in the order of importance
        (See _get_sorted_bond_tensor for the importance metric)
        """
        #sort bond_dict into bond_tensor
        bond_dict            = json.loads(row_dict['bonds_json'])
        sorted_bond_tensor   = self._get_sorted_bond_tensor(bond_dict)

        #Get number of atoms and initialize each variable
        n_atoms              = row_dict['num_atoms']
        bond_property_tensor = torch.zeros(self.num_atoms_after_padding,self.filter_length,2)
        count                = torch.ones(self.num_atoms_after_padding).int()
        ind_arrays           = torch.zeros(self.num_atoms_after_padding,self.filter_length).int()

        #Iterate through the bonds and add them
        for (fromNode, toNode, dis, bondorder) in sorted_bond_tensor:
            fromNode, toNode = int(fromNode), int(toNode)
            if bondorder>bond_order_cutoff and count[fromNode]<self.filter_length-1:
                #Convert float to int
                bond_property_tensor[fromNode][count[fromNode]] = torch.Tensor([dis, bondorder])
                ind_arrays[fromNode][count[fromNode]] = toNode
                count[fromNode] += 1

        #Create connectivity_tensor
        #Shape of Tensor is n_atoms by n_atoms by filter_length
        connectivity_tensor  = torch.zeros(self.num_atoms_after_padding,self.num_atoms_after_padding,self.filter_length)
        for atom_ind, ind_array in enumerate(ind_arrays):
            clipped_ind_array       = ind_array[:count[atom_ind]]
            #Add the index of the current atom to top of array
            clipped_ind_array[0]    = atom_ind

            if atom_ind < n_atoms:
                connectivity_tensor[atom_ind] = self.get_one_hot(clipped_ind_array, self.num_atoms_after_padding, self.filter_length)
            else:
                connectivity_tensor[atom_ind] = torch.zeros(self.num_atoms_after_padding,self.filter_length)
        return connectivity_tensor, bond_property_tensor

    @staticmethod
    def get_one_hot(atom_index_vector, N, filter_length):
        # N is number of atoms in this case
        output = torch.zeros(N,filter_length)
        for (location, atom_index) in enumerate(atom_index_vector):
            output[atom_index][location] += 1
        return output


def extract_raw_data(output_dir, overwrite = False, limit = None):
    """Extracts all of the raw data from sqlite3 database, transforms it into
    Net inputs then saves it to output_dir"""
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print 'Warning: {} already exists'.format(output_dir)

    #Initialize extractor
    extractor = RawDataExtractor(limit = limit)
    material_ids = extractor.material_ids

    #If we are not overwriting existing raw_inputs remove existing material_ids
    if not overwrite:
        extracted_material_ids = os.listdir(output_dir)
        material_ids = filter(lambda id_curr: not id_curr in extracted_material_ids, material_ids)

    #Iterate through extractor and save the tuple of sample_input and sample_label
    #to file in output_dir with the name material_id.torch
    for material_id  in tqdm(material_ids):
        if not os.path.exists(os.path.join(output_dir,material_id+'.torch')) or overwrite:
            sample_data = extractor[material_id]
            torch.save(sample_data,os.path.join(output_dir, material_id+'.torch'))

def split_test_set(data_dir, output_dir, test_split = 0.2, val_split = 0.2):
    """splits training+dev set from the test set, test_split is the fraction of
    samples in the test set Values are saved to output_dir/test"""
    filenames = os.listdir(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.torch')]
    filenames.sort()
    seed = 42
    train_and_val_filenames, test_filenames = train_test_split(filenames,test_size = test_split, random_state = seed)
    train_filenames, val_filenames = train_test_split(train_and_val_filenames,test_size = val_split, random_state = seed+1)

    filenames = {'train': train_filenames,
                 'val': val_filenames,
                 'test': test_filenames}

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("Warning: output dir {} already exists".format(output_dir))

    # Preprocess train, val and test
    for split in ['train', 'val', 'test']:
        output_dir_split = os.path.join(output_dir, '{}'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            shutil.copy(filename,os.path.join(output_dir_split,os.path.basename(filename)))

    print("Done building dataset")

def build_tiny_dataset(limit = 10):
    data_dir    = os.path.join(datasets_folder,'tiny_raw_input')
    dataset_dir = os.path.join(datasets_folder,'tiny_dataset')
    extract_raw_data(data_dir,overwrite = True, limit = limit)
    split_test_set(data_dir, dataset_dir)



if __name__ == '__main__':
    args = parser.parse_args()
    # build_tiny_dataset()
    # extract_raw_data(args.data_dir, overwrite = True, limit = None)
    split_test_set(args.data_dir,args.output_dir)
