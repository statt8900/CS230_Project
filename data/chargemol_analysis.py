#External Modules
import sql,os,subprocess,json,shutil
import numpy as np
import collections

from ase.io import write, read
import ase
from CataLog.chargemol.chargemol import BondAnalyzer

#Internal Modules
from    CS230_Project.misc.sql_shortcuts import *
from    CS230_Project.misc.utils         import safeMkDir, check_if_on_sherlock, traj_rebuild, flatten, negate, print_time
import  CS230_Project.data.database_management as db

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
loaded_chargemol_folder = os.environ['CS230_finished_chargemol_folder']

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
            os.chdir(pth)
            BondAnalyzer(dftcode='gpaw',quality='low').submit(pth,'final')


def get_running_materials_ids():
    """ Get list of material_ids for currently running chargemol analysis"""
    all_current_jobs            = subprocess.check_output(['squeue','-o','%Z']).split('\n')
    jobs_in_chargemol_folder    = filter(lambda dir_curr: chargemol_folder in dir_curr, all_current_jobs)
    currently_running_materials_id = [x.replace(chargemol_folder,'',1).split('/')[1] for x in jobs_in_chargemol_folder]
    return currently_running_materials_id

def get_failed_material_ids():
    """ Get list of material_ids for failed chargemol analysis"""
    finished_ids =  db.Query(constraints = [PMG_Entries.chargemol]).query_col(PMG_Entries.material_id)
    attempted_ids = os.listdir(chargemol_folder)
    failed_ids = filter(lambda id_curr: not id_curr in finished_ids, attempted_ids)
    return failed_ids

def move_finished_material_ids():
    """ Move the finished directories from the running directory to the finished folder"""
    finished_ids =  db.Query(constraints = [PMG_Entries.chargemol]).query_col(PMG_Entries.material_id)
    attempted_ids = os.listdir(chargemol_folder)
    loaded_ids = os.listdir(loaded_chargemol_folder)
    failed_ids = filter(lambda id_curr: not id_curr in finished_ids, attempted_ids)
    succesful_ids_that_need_to_be_moved = filter(lambda id_curr: id_curr in finished_ids and id_curr not in loaded_ids, attempted_ids)
    for id_curr in succesful_ids_that_need_to_be_moved:
            try:
                old_pth = os.path.join(chargemol_folder,id_curr)
                new_pth = os.path.join(loaded_chargemol_folder,id_curr)
                shutil.move(old_pth, new_pth)
                print 'Moving {} from {} to {}'.format(id_curr, old_pth,new_pth)
            except OSError:
                print 'Failed to move {}'.format(id_curr)

def unpack_directories():
    """unpack the nested directories from chargemol"""
    attempted_ids   = os.listdir(chargemol_folder)
    running_ids     = get_running_materials_ids()
    not_running_ids = filter(lambda id_curr: not id_curr in running_ids, attempted_ids)
    succesful_ids   = filter(lambda id_curr: os.path.exists(os.path.join(chargemol_folder
                                                                        ,id_curr
                                                                        ,'chargemol_analysis'
                                                                        ,'final'
                                                                        ,'bonds.json')) ,not_running_ids)
    for id_curr in succesful_ids:
        print id_curr
        os.system('mv '+os.path.join(chargemol_folder,id_curr,'chargemol_analysis','final','*')+ ' '+os.path.join(chargemol_folder,id_curr))
        os.system('rm -r '+os.path.join(chargemol_folder,id_curr,'chargemol_analysis'))

def change_permissions():
    """Change permmissions on all chargemol folders"""
    from os import stat
    from pwd import getpwuid

    def find_owner(filename):
        return getpwuid(stat(filename).st_uid).pw_name

    attempted_ids   = os.listdir(chargemol_folder)

    for id_curr in attempted_ids:
        dir_curr = os.path.join(chargemol_folder,id_curr)
        if find_owner(dir_curr) == os.environ['USER']:
            print 'Changing permmissions on {}'.format(dir_curr)
            os.chmod(dir_curr, 0777)

def load_chargemol():
    unpack_directories()
    db.update_chargemol()
    move_finished_material_ids()
