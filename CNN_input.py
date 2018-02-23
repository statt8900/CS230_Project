#External Modules
import os, ase, json, ase.io, collections
import numpy as np
import pdb
import torch
from torch.utils.data import Dataset, DataLoader

#Internal Modules
import CS230_Project.data.database_management as db
from CS230_Project.misc.sql_shortcuts import *
from CS230_Project.misc.utils import traj_rebuild

necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

#Sorting Functions
bo_over_dis = lambda distance, bondorder: bondorder/distance

class CNNInputDataset(Dataset):
    """
    Subclass of Torch Dataset object
    storage_directories :: list of storage directories where chargemol analysis,
                            result.json, and final.traj are stored
    """
    def __init__(self,constraints = [], limit = 10, filter_length = 13):
        default_constraints = [PMG_Entries.chargemol]
        constraints += default_constraints
        self.query = db.Query(constraints = constraints, limit = limit)
        self.output_dict = self.query.query_dict(cols = ['*'])
        self.filter_length = filter_length

    def __getitem__(self, index):
        return self.row_dict_to_CNN_input(self.output_dict[index])

    def __len__(self):
        return len(self.output_dict)

    def row_dict_to_CNN_input(self, row_dict):
        """
        Take a storage directory, with chargemol_analysis and job_output subfolders,
        and produce a connectivity matrix
        """
        e_form              = row_dict['formation_energy_per_atom']
        #Extract the connectivity
        connectivity_tensor, bond_property_tensor = self.row_dict_to_connectivity_tensors(row_dict)
        #Get the node feature matrix
        atoms_obj           = traj_rebuild(row_dict['atoms_obj'])
        node_feature_matrix = self.atoms_to_node_features(atoms_obj)
        return (node_feature_matrix, connectivity_tensor, bond_property_tensor, e_form)

    def atoms_to_node_features(self, atoms):
        """
        Converts atoms object into a numpy array
        node_feature_matrix is shape (num_atoms,number_of_features)

        TO CHANGE: number_of_features = 2 (period,group)
        """

        node_feature_matrix = torch.zeros(len(atoms),2)
        for (i,atom) in enumerate(atoms):
            node_feature_matrix[i] = self.get_atom_features(atom)
        return node_feature_matrix

    def get_atom_features(self, atom):
        """
        Returns numpy array of atom get_atom_features
        1st iteration: Feature for 1 atom is [period, group]
        """
        period = [0,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3
                  ,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4
                  ,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
                  ,6,6,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6
                  ,7,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]

        group =   [0,1,18,1,2,13,14,15,16,17,18,1,2,13,14,15,16,17,18
                  ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
                  ,1,2,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
        atomic_num = atom.number
        return torch.Tensor([period[atomic_num],group[atomic_num]])

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
        unpacked_bond_dict = [[bond['fromNode'],bond['toNode'],bond['distance'],bond['bondorder']] for bond in bond_dict]
        bond_tensor = torch.Tensor(unpacked_bond_dict)

        #Apply bond function and sort the bond_tensor by that value
        bond_sort_value = torch.Tensor(map(bond_function,*zip(*bond_tensor[:,2:])))
        _, sort_indices = torch.sort(bond_sort_value, descending = True)
        sorted_bond_tensor = bond_tensor[sort_indices,:]
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
        node_feature_matrix to create a matrix that can be convulved with a filter


        bond_property_tensor    :: Type     = pytorch tensor
                                :: Shape    = (n_atoms, filter_length, 2)
        Description: Each atom gets a filter_length by 2 tensor that has bond properties
        (distance, bondorder) listed in the order of importance
        (See _get_sorted_bond_tensor for the importance metric)
        """
        #sort bond_dict into bond_tensor
        bond_dict           = json.loads(row_dict['bonds_json'])
        sorted_bond_tensor  = self._get_sorted_bond_tensor(bond_dict)

        #Get number of atoms and initialize each variable
        n_atoms = row_dict['num_atoms']
        #n_atoms
        bond_property_tensor = torch.zeros(n_atoms,self.filter_length,2)
        count = torch.ones(n_atoms).int()
        ind_arrays = torch.zeros(n_atoms,self.filter_length).int()

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
        connectivity_tensor = torch.zeros(n_atoms,n_atoms,self.filter_length)
        for atom_ind, ind_array in enumerate(ind_arrays):
            clipped_ind_array       = ind_array[:count[atom_ind]]
            #Add the index of the current atom to top of array
            clipped_ind_array[0]    = atom_ind
            connectivity_tensor[atom_ind] = self.get_one_hot(clipped_ind_array, n_atoms, self.filter_length)

        return connectivity_tensor, bond_property_tensor


    @staticmethod
    def get_one_hot(atom_index_vector, N, filter_length):
        # N is number of atoms in this case
        output = torch.zeros(N,filter_length)
        for (location, atom_index) in enumerate(atom_index_vector):
            output[atom_index][location] += 1
        return output
