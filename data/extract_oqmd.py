import os
import matplotlib.pyplot as plt
import MySQLdb
import torch
import numpy as np
from copy import deepcopy

db = MySQLdb.connect("localhost","root","@L@ftlin3","oqmd2")
cursor = db.cursor()


### Theory experiment comparison ################
# command = '''
#
# select
#     theory.composition_id,
#     min(expt.delta_e) as e_expt,
#     min(theory.delta_e) as e_theory
#
# from
#     (select composition_id, delta_e from expt_formation_energies) as expt,
#     formation_energies as theory
#
# where
#     theory.composition_id = expt.composition_id
#
# group by
#     theory.composition_id
#
# limit 10000
#
# '''
#
# cursor.execute(command)
# results = cursor.fetchall()
#
# for result in results:
#     plt.plot(result[1],result[2],'o')
#
# plt.show()
####################################################

def get_one_hot(atom_index_vector, N, filter_length):
    # N is number of atoms in this case
    output = torch.zeros(N,filter_length)
    for (location, atom_index) in enumerate(atom_index_vector):
        output[atom_index][location] += 1
    return output

def fractional_to_actual(fractional_positions, x,y,z):
    n_atoms = fractional_positions.shape[0]
    x_comp = fractional_positions[:,0].repeat(3).reshape(n_atoms,3) * np.tile(x,n_atoms).reshape(n_atoms,3)
    y_comp = fractional_positions[:,1].repeat(3).reshape(n_atoms,3) * np.tile(y,n_atoms).reshape(n_atoms,3)
    z_comp = fractional_positions[:,2].repeat(3).reshape(n_atoms,3) * np.tile(z,n_atoms).reshape(n_atoms,3)

    positions = x_comp + y_comp + z_comp

    return positions

def get_repeated_unit_cell(cursor,structure_id):
    cursor.execute("select x1,x2,x3,y1,y2,y3,z1,z2,z3 from structures where id = {}".format(structure_id))
    (x1,x2,x3,y1,y2,y3,z1,z2,z3) = cursor.fetchall()[0]
    x = np.array([x1,x2,x3])
    y = np.array([y1,y2,y3])
    z = np.array([z1,z2,z3])

    cursor.execute("select id,x,y,z,fx,fy,fz from atoms where structure_id = {}".format(structure_id))
    positions_data = cursor.fetchall()
    fractional_positions = np.vstack([np.array([row[1],row[2],row[3]]) for row in positions_data])
    forces = np.vstack([np.array([row[4],row[5],row[6]]) for row in positions_data])
    n_atoms = fractional_positions.shape[0]

    positions = fractional_to_actual(fractional_positions, x,y,z)

    repeated_positions = np.zeros((27*n_atoms,3))

    oqmd_id_lookup = np.array([tup[0] for tup in positions_data])
    ids = np.arange(n_atoms)
    repeated_ids = np.tile(ids,27)

    count = 0
    for i in [0,1,-1]:
        for j in [0,1,-1]:
            for k in [0,1,-1]:
                repeated_positions[count*n_atoms:(count+1)*n_atoms,:] = positions + i*x + j*y + k*z
                count += 1

    return repeated_positions, repeated_ids, oqmd_id_lookup, forces


def get_bond_property_tensor(repeated_positions, repeated_ids, filter_length, A):
    # A is max number of atoms
    n_atoms = int(repeated_ids.shape[0]/27)
    bond_property_tensor = torch.zeros((A, filter_length))
    bond_property_tensor_xyz = torch.zeros((A, filter_length, 4))
    connectivity_tensor = torch.zeros((A,A,filter_length))
    for a in range(n_atoms):
        xyz = deepcopy(repeated_positions)
        current_ids = deepcopy(repeated_ids)
        xyz -= xyz[a,:]
        square_dists = np.sum(xyz**2., axis=1)
        id_sq_dist_pairs = np.vstack((current_ids, square_dists, xyz.T)).T
        sorted_pairs = id_sq_dist_pairs[id_sq_dist_pairs[:,1].argsort()][0:filter_length,:]
        bond_property_tensor[a,:] = torch.FloatTensor(sorted_pairs[:,1])
        bond_property_tensor_xyz[a,:,:] = torch.FloatTensor(sorted_pairs[:,1:])

        one_hot = get_one_hot(sorted_pairs[:,0], n_atoms, filter_length)
        connectivity_tensor[a,0:n_atoms,:] = one_hot

    mask_atom_tensor = torch.zeros(A)
    mask_atom_tensor[0:n_atoms] = 1

    return bond_property_tensor, bond_property_tensor_xyz, connectivity_tensor, mask_atom_tensor


A = 50
n_input_features = 12
filter_length = 13
save_dir = '/Users/brohr/Documents/Stanford/Research/scripts/ML/ChemConvData/Datasets/data_storage/oqmd'

cursor.execute("select entry_id, structures.id as structure_id from structures join formation_energies using(entry_id) where structures.label = 'final' and formation_energies.delta_e is not null and (select count(*) from atoms where atoms.structure_id = structures.id) <= {} limit 2".format(A))
entry_structure_ids = cursor.fetchall()
# entry_structure_ids = ((5810, 34576), (5810, 34576))

all_structure_ids = torch.zeros(len(entry_structure_ids))
for i, (entry_id, structure_id) in enumerate(entry_structure_ids):
    print i, entry_id, structure_id
    all_structure_ids[i] = structure_id
    cursor.execute("select elements.group,period,atomic_radii,covalent_radii,melt,boil,electronegativity,first_ionization_energy, s_elec,p_elec,d_elec,f_elec from atoms join elements on elements.symbol = atoms.element_id where atoms.structure_id = {} order by atoms.id".format(structure_id))
    node_prop_data = cursor.fetchall()
    node_property_tensor = torch.zeros(A,n_input_features)
    for j, row in enumerate(node_prop_data):
        node_property_tensor[j,:] = torch.FloatTensor(row)
    torch.save(node_property_tensor, os.path.join(save_dir, str(structure_id) + '_node_property_tensor.torch'))

    cursor.execute("select delta_e from formation_energies where entry_id = {}".format(entry_id))
    e_form = float(cursor.fetchall()[0][0])
    torch.save(e_form*torch.ones(1), os.path.join(save_dir, str(structure_id) + '_e_form.torch'))

    cursor.execute("select fx,fy,fz from atoms where structure_id = {}".format(structure_id))

    repeated_positions, repeated_ids, oqmd_id_lookup, forces = get_repeated_unit_cell(cursor,structure_id)
    np.save(open(os.path.join(save_dir, str(structure_id) + '_oqmd_id_lookup.npy'),'wb'), oqmd_id_lookup)
    forces_target = np.zeros((A,3))
    forces_target[0:forces.shape[0],:] = forces
    torch.save(torch.from_numpy(forces_target), os.path.join(save_dir, str(structure_id) + '_forces_target.torch'))
    bond_property_tensor, bond_property_tensor_xyz, connectivity_tensor, mask_atom_tensor = get_bond_property_tensor(repeated_positions, repeated_ids, filter_length, A)
    torch.save(bond_property_tensor, os.path.join(save_dir, str(structure_id) + '_bond_property_tensor.torch'))
    torch.save(bond_property_tensor_xyz, os.path.join(save_dir, str(structure_id) + '_bond_property_tensor_xyz.torch'))
    torch.save(connectivity_tensor, os.path.join(save_dir, str(structure_id) + '_connectivity_tensor.torch'))
    torch.save(mask_atom_tensor, os.path.join(save_dir, str(structure_id) + '_mask_atom_tensor.torch'))

torch.save(all_structure_ids, os.path.join(save_dir, 'all_structure_ids.torch'))















































#let me scroll
