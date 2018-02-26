#External Modules
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from ase.data import chemical_symbols
import pickle, os, sqlite3
from copy import deepcopy

#Internal modules
from CS230_Project.data.database_management import load_row_dictionary
from CS230_Project.misc.utils import traj_preparer

################################################################################
"""This module contains functions to query the Materials Project to generate our
dataset"""
################################################################################

#Check for required enviromental variables
necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'


###########################
# Materials Project Queries
#--------------------------

def query_all_of_materials_project():
    """
    Query all of the materials project in single element queries. Combines all the
    queries and writes the output to a pickle file
    """

    #Initialize the API query object
    API_key = 'mhaSG6L5b4HOQv2A'
    mp_rest = MPRester(API_key)

    #properties useful for project
    useful_properties =   [u'energy'
                            ,u'band_gap'
                            ,u'e_above_hull'
                            ,u'elements'
                            ,u'pretty_formula'
                            ,u'formation_energy_per_atom'
                            ,u'cif'
                            ,u'density'
                            ,u'material_id'
                            ,u'nelements'
                            ,u'energy_per_atom']
    #Other Possible Properties
    #Full description of each property at
    #https://materialsproject.org/docs/api#materials_.28calculated_materials_data.29
                            # ,u'is_compatible'
                            # ,u'elasticity'
                            # ,u'unit_cell_formula'
                            # ,u'oxide_type'
                            # ,u'hubbards'
                            # ,u'task_ids'
                            # ,u'nsites'
                            # ,u'icsd_id'
                            # ,u'tags'
                            # ,u'volume'
                            # ,u'total_magnetization'
                            # ,u'is_hubbard'
                            # ,u'spacegroup'
                            # ,u'full_formula'

    #Divide the elements into groups of 1 to make small queries so PMG doesn't
    #get mad
    num_eles = len(chemical_symbols)
    divided = num_eles
    criteria_list = [{'elements':{'$all':chemical_symbols[x*num_eles/divided:num_eles*(x+1)/divided]}} for x in range(divided)]

    #For each criteria query the materials project and append the output
    query_output = []
    for criteria in criteria_list:
        print 'Querying Elements in Criteria : {0}'.format(criteria)
        query_output += mp_rest.query(criteria, useful_properties)

    #Need to remove any duplicate jobs (i.e. jobs with same material_id)
    unique_query_output = []
    counted_mat_ids = []
    for row in query_output:
        if not row['material_id'] in counted_mat_ids:
            unique_query_output.append(row)
            counted_mat_ids.append(row['material_id'])

    #Write the output to a pickle file
    project_folder = os.environ['CS230_Project_Folder']
    with open(project_folder+'/data/raw_data/mat_proj_query.pickle','w') as file_curr:
        file_curr.write(pickle.dumps(new_output))

    return unique_query_output


def convert_cif_to_ase(cif_string):
    """
    Convert PMG cif string format to an ase Atoms object
    """
    structure = CifParser.from_string(cif_string).get_structures()[-1]
    return AseAtomsAdaptor.get_atoms(structure)

def read_mat_proj_pickle():
    return pickle.load(open(os.environ['CS230_Project_Folder']+'/data/raw_data/mat_proj_query.pickle'))

def load_database_from_mat_proj_pickle():
    """
    1. Read in the pickle file output of query_all_of_materials_project()
    2. Convert each cif file to an ase Atoms object
    3. load each new row into the database
    """

    new_output = []
    output = read_mat_proj_pickle()
    for i, row in enumerate(output):
        if i%1000 == 0:
            print i
        new_row = deepcopy(row)
        atoms_obj = convert_cif_to_ase(row['cif'])
        new_row['atoms_obj'] = traj_preparer(atoms_obj)
        new_row['num_atoms'] = len(atoms_obj)
        new_row['chargemol'] = 0
        new_row['bond_json'] = None
        new_output.append(new_row)
        try:
            load_row_dictionary(new_row)
        except sqlite3.OperationalError:
            pass
