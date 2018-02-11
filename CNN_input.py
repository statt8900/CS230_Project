#External Modules
import os, ase, json, ase.io
import numpy as np
import pdb
from torch.utils.data import Dataset
#Internal Modules
import CS230_Final_Project.data.database_management as db
from CS230_Final_Project.misc.sql_shortcuts import *
from CS230_Final_Project.misc.utilities import traj_rebuild

necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

class CNNInputDataset(Dataset):
    """
    Subclass of Torch Dataset object
    storage_directories :: list of storage directories where chargemol analysis,
                            result.json, and final.traj are stored
    """
    def __init__(self,constraints = [], limit = 10):
        default_constraints = [PMG_Entries.chargemol]
        constraints += default_constraints
        self.query = db.Query(constraints = constraints, limit = limit)
        self.output_dict = self.query.query_dict(cols = ['*'])

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
        connectivity        = self.row_dict_to_connectivity(row_dict)
        #Get the node feature matrix
        atoms_obj           = traj_rebuild(row_dict['atoms_obj'])
        node_feature_matrix = self.atoms_to_node_features(atoms_obj)
        return (connectivity, node_feature_matrix, e_form)

    def atoms_to_node_features(self, atoms):
        """
        Converts atoms object into a numpy array
        node_feature_matrix is shape (num_atoms,number_of_features)

        TO CHANGE: number_of_features = 2 (period,group)
        """

        node_feature_matrix = np.zeros((len(atoms),2))
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
        return np.array([period[atomic_num],group[atomic_num]])

    def row_dict_to_connectivity(self, row_dict):
        """
        Uses bonds.json to produce a list, with each element being a list of
        pairs (toAtomIndex, bond order)
        """
        output = []
        bond_dict = json.loads(row_dict['bonds_json'])
        n = row_dict['num_atoms']
        for i in range(n):
            newatom = []
            for bond in bond_dict:
                if bond['fromNode'] == i:
                    if bond['bondorder'] > 0.01:
                        newatom.append((bond['bondorder'],bond['distance'],bond['toNode']))

            sorted_newatom = list(reversed(sorted(newatom))) # bonds in decreasing strength
            maxind = min(12,len(newatom))              # take UP TO 12 bonds
            out_list = [(n,b,d) for b,d,n in sorted_newatom[:maxind]]
            output.append(np.array(out_list))
        return output
