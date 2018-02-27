#External Modules
import sql,os, subprocess,json
import numpy as np
import collections

from ase.io import write, read
import ase

#Internal Modules
from CS230_Project.misc.sql_shortcuts import *
from CS230_Project.misc.utils     import safeMkDir, check_if_on_sherlock, traj_rebuild, flatten, negate, print_time
import CS230_Project.data.database_management as db

################################################################################
"""
This module contains functions needed for preprocessing database structures
such that bond strengths are available as input data.
"""
################################################################################

#Check for required enviromental variables
necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

#Set enviromental variables for required enviromental variables
project_folder = os.environ['CS230_Project_Folder']
chargemol_folder = os.environ['CS230_chargemol_folder']


def meta_bond_analyze(constraints = [], limit = 20):
    """
    !!!Can Only be run on sherlock currently to prevent database conflicts!!!!

    Take a random PMG entry in database and run chargemol on it if it hasn't
    been done before
    constraints :: List of python-sql constraints to be added to the default constraints
    limit       :: Int, number of jobs to be randomly sampled
    """
    from CataLog.chargemol.chargemol                 import BondAnalyzer
    assert check_if_on_sherlock, 'Can only run chargemol on sherlock'
    db.update_chargemol()
    running_ids = get_running_materials_ids()
    failed_ids = get_failed_material_ids()
    # default_constraints = [PMG_Entries.chargemol==0, NotIn(PMG_Entries.material_id, running_ids)]
    constraints = [Not(PMG_Entries.chargemol)]

    query = db.Query(constraints = constraints
                    , cols = [PMG_Entries.material_id
                             ,PMG_Entries.atoms_obj]
                    , order = Random()
                    , limit = limit
                    , verbose = True)
                    
    mat_ids, atoms_obj_pickle = zip(*query.query())
    atoms_obj_list = map(traj_rebuild, atoms_obj_pickle)
    for mat_id, atoms_obj  in zip(mat_ids,atoms_obj_list):
        if not mat_id in failed_ids and not mat_id in running_ids:
            pth = os.path.join(chargemol_folder,mat_id)
            safeMkDir(pth)
            os.chmod(pth,0755)
            if not os.path.exists(pth+'/final.traj'):
                write(pth+'/final.traj',atoms_obj)
            BondAnalyzer().submit(pth,'final')
            #os.chdir(pth)
            #submit_script(pth,pth+'/final.traj')


def get_running_materials_ids():
    """ Get list of material_ids for currently running chargemol analysis"""
    all_current_jobs            = subprocess.check_output(['squeue','-o','%Z']).split('\n')
    jobs_in_chargemol_folder    = filter(lambda dir_curr: chargemol_folder in dir_curr, all_current_jobs)
    currently_running_materials_id = map(os.path.basename, jobs_in_chargemol_folder)
    return currently_running_materials_id

def get_failed_material_ids():
    """ Get list of material_ids for failed chargemol analysis"""
    finished_ids =  db.Query(constraints = [PMG_Entries.chargemol]).query_col(PMG_Entries.material_id)
    attempted_ids = os.listdir(chargemol_folder)
    failed_ids = filter(lambda id_curr: not id_curr in finished_ids, attempted_ids)
    return failed_ids
