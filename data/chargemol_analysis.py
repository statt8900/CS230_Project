#External Modules
import sql,os
from chargemol                              import submit_script
from ase.io import write
#Internal Modules
from CS230_Final_Project.misc.sql_shortcuts import *
from CS230_Final_Project.misc.utilities     import safeMkDir, check_if_on_sherlock, traj_rebuild
import CS230_Final_Project.data.database_management as db


necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

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
    assert check_if_on_sherlock, 'Can only run chargemol on sherlock'
    fin_dirs = db.update_chargemol()
    default_constraints = [PMG_Entries.chargemol==0, NotIn(PMG_Entries.material_id, fin_dirs)]
    constraints += default_constraints

    query = db.Query(constraints = constraints, cols = [PMG_Entries.material_id, PMG_Entries.atoms_obj], order = Random(), limit = limit)
    mat_ids, atoms_obj_pickle = zip(*query.query())
    atoms_obj_list = map(traj_rebuild, atoms_obj_pickle)

    for mat_id, atoms_obj  in zip(mat_ids,atoms_obj_list):
        pth = os.path.join(chargemol_folder,mat_id)
        safeMkDir(pth)
        os.chmod(pth,0755)
        if not os.path.exists(pth+'final.traj'):
            write(pth+'/final.traj',atoms_obj)
        os.chdir(pth)
        submit_script(pth,pth+'/final.traj')
