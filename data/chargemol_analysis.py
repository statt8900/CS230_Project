#External Modules
import sql,os, subprocess,json
import networkx as nx
import numpy as np
import collections
from chargemol                              import submit_script
from ase.io import write, read
import ase
#Internal Modules
from CS230_Project.misc.sql_shortcuts import *
from CS230_Project.misc.utils     import safeMkDir, check_if_on_sherlock, traj_rebuild, flatten, negate, print_time
import CS230_Project.data.database_management as db


necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

project_folder = os.environ['CS230_Project_Folder']
chargemol_folder = os.environ['CS230_chargemol_folder']

# time_dict = {'low':.5,'mid':1,'high':2}
#
# def submit_chargemol(pth,traj=None,code='gpaw',quality='low'):
#     sub_pth = os.path.join(pth,'sub_chargemol.')
#
#     if traj is None:
#         traj = glob.glob(os.path.join(pth,'*.traj'))[0]
#
#     subpy ='\n'.join(['import chargemol'
#                     ,"chargemol.bond_analyze('%s','%s','%s','%s')\n"%(pth,traj,code,quality)])
#
#     subsh = '\n'.join(['#!/bin/bash'
#                     ,'#SBATCH -p iric,owners'
#                     ,'#SBATCH --time=%s:00'%(print_time(time_dict[quality]))
#                     ,'#SBATCH --mem-per-cpu=4000'
#                     ,'#SBATCH --error=err.log'
#                     ,'#SBATCH --output=opt.log'
#                     ,'#SBATCH --nodes=1'
#                     ,'#SBATCH --ntasks-per-node=16'
#                     ,"NTASKS=`echo $SLURM_TASKS_PER_NODE|tr '(' ' '|awk '{print $1}'`"
#                     ,"NNODES=`scontrol show hostnames $SLURM_JOB_NODELIST|wc -l`"
#                     ,'NCPU=`echo " $NTASKS * $NNODES " | bc`'
#                     ,'source /scratch/users/ksb/gpaw/paths.bash'
#                     ,'mpirun -n $NCPU gpaw-python sub_chargemol.py'])
#     with open(sub_pth+'sh','w') as f: f.write(subsh)
#     with open(sub_pth+'py','w') as f: f.write(subpy)
#     if not os.path.exists(os.path.join(pth,'bonds.json')):
#         os.chdir(pth)
#         map(lambda file_curr: os.chmod(file_curr,0777), os.walk(os.getcwd()).next()[2])
#         os.system('sbatch sub_chargemol.sh')
#


def meta_bond_analyze(constraints = [], limit = 20):
    """
    !!!Can Only be run on sherlock currently to prevent database conflicts!!!!

    Take a random PMG entry in database and run chargemol on it if it hasn't
    been done before
    constraints :: List of python-sql constraints to be added to the default constraints
    limit       :: Int, number of jobs to be randomly sampled
    """
    assert check_if_on_sherlock, 'Can only run chargemol on sherlock'
    db.update_chargemol()
    running_ids = get_running_materials_ids()
    failed_ids = get_failed_material_ids()
    # default_constraints = [PMG_Entries.chargemol==0, NotIn(PMG_Entries.material_id, running_ids)]
    constraints = [PMG_Entries.chargemol==0]

    query = db.Query(constraints = constraints, cols = [PMG_Entries.material_id, PMG_Entries.atoms_obj], order = Random(), limit = limit)
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
            submit_script(pth,pth+'/final.traj')


def get_running_materials_ids():
    all_current_jobs            = subprocess.check_output(['squeue','-o','%Z']).split('\n')
    jobs_in_chargemol_folder    = filter(lambda dir_curr: chargemol_folder in dir_curr, all_current_jobs)
    currently_running_materials_id = map(os.path.basename, jobs_in_chargemol_folder)
    return currently_running_materials_id

def get_failed_material_ids():
    finished_ids =  db.Query(constraints = [PMG_Entries.chargemol]).query_col(PMG_Entries.material_id)
    attempted_ids = os.listdir(chargemol_folder)
    failed_ids = filter(lambda id_curr: not id_curr in finished_ids, attempted_ids)
    return failed_ids

class Edge(object):
    def __init__(self,fromNode,toNode,distance,offset,bondorder,vector=None):
        self.fromNode   = fromNode
        self.toNode     = toNode
        self.weight     = distance
        self.pbc_shift  = map(int,(offset[0],offset[1],offset[2]))
        self.bondorder  = bondorder
        self.vector     = vector

    def __str__(self): return "<Edge(%d,%d,%.2f A,%s,bo: %.2f)>"%(
                                self.fromNode,self.toNode,self.weight,self.pbc_shift
                                ,self.bondorder)

    def __repr__(self): return str(self)

    def add_edge_to_graph(self,grph):
        """Takes a networkx.Graph object, adds itself to it"""
        prop_dict = {k: self.__dict__[k] for k in ('weight', 'pbc_shift', 'bondorder')}
        tupl = (self.toNode,self.fromNode)
        for (i,j,d) in grph.edges(data=True):
            if ((i,j),d['pbc_shift'])==(tupl,map(negate,self.pbc_shift)):
                return None # REPEAT EDGE, DON'T ADD
        grph.add_edges_from([tupl],**prop_dict)

    def inds_pbc(self): return (self.fromInd,self.toInd,self.pbc_shift)


class GraphMaker(object):
    """
    Interface between Chargemol output (bond.json) and graph representations
    include_frac  :: Float
    group_cut     :: Float
    min_bo        :: Float
    jmol          :: Float
    """
    def __init__(self, include_frac=0.8,group_cut=0.3,min_bo=0.03,jmol=False):
        self.include_frac   = include_frac
        self.group_cut      = group_cut
        self.min_bo         = min_bo
        self.colored        = True
        self.jmol           = 0.01 if jmol==True else jmol # give a FLOAT to specify jmol tolerance

    def _get_edge_dicts(self,material_id,trajname):
        """Finds bonds.json, loads it as a list of dictionaries"""
        jsonpth = os.path.join(chargemol_folder,material_id,'bonds.json')
        with open(jsonpth,'r') as f: return json.load(f)

    def _make_edges(self,material_id,trajname):
        """Takes dejson'd bonds.json and makes Edge objects from the list"""
        edges = collections.defaultdict(list)

        if self.jmol:
            atoms = self.make_atoms(material_id,trajname)
            jmol_output = jmol(atoms,self.jmol)
            for e in jmol_output:
                edges[e.fromNode].append(e)
            return edges
        else:
            dicts = self._get_edge_dicts(material_id,trajname)
            keys  = ['distance','bondorder','fromNode','toNode','offset']
            for d in dicts:
                d,bo,i,j,o = [d[x] for x in keys]
                if bo > self.min_bo:
                    edges[i].append(Edge(i,j,d,np.array(o),bo)) # defaultdict useful
            return edges

    def _make_adjacency_matrix(self,material_id,trajname):
        """
        Returns an n_atoms by n_atoms matrix where each row column pair contains
        the sum of the bond orders for the edges between those  two atoms
        """
        edge_dict = self._make_edges(material_id,trajname)
        atoms = self.make_atoms(material_id,trajname)
        output = np.zeros((len(atoms),len(atoms)))

        for i,edges in edge_dict.items():
            for edge in edges: output[i,edge.toNode] += edge.bondorder
        return output

    def _make_group(self,edge_list, group_cut = None):
        from scipy.cluster.hierarchy import fclusterdata
        if group_cut is None: group_cut = self.group_cut

        # Handle edge cases
        if not edge_list:      return edge_list
        if len(edge_list)==1:  return [edge_list]

        strs = np.array([[e.bondorder for e in edge_list]]).T # create (n_edges,1) array
        groups = collections.defaultdict(list)    # initialize group dict

        group_inds = list(fclusterdata(strs,group_cut
                                ,criterion='distance',method='ward'))

        for i in range(len(edge_list)): groups[group_inds[i]].append(edge_list[i])
        maxbo_groups = [(max([e.bondorder for e in es]),es) for es in groups.values()]
        sorted_maxbo_groups = list(reversed(sorted(maxbo_groups)))
        return [es for maxbo,es in sorted_maxbo_groups]


    def _get_filtered_edges(self,material_id,trajname): # BOA
        """
        Return a filtered subset of the edges serialized in bonds.json
        """

        output   = []                                   # initialize
        edges    = self._make_edges(material_id,trajname)   # {index:list of edges from index}
        if self.jmol:
            for e in  flatten(edges.values()):    # only need to filter duplicate edges
                dup_e = Edge(e.toNode,e.fromNode,e.weight,map(negate,e.pbc_shift),e.bondorder)
                if dup_e not in output: output.append(e)
            return output
        total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}
        groups   = {ind: self._make_group(es) for ind,es in edges.items()}

        max_ind  = max([max(e.fromNode, e.toNode) for e in flatten(edges.values())])

        for ind in range(max_ind+1):
            accumulated = 0
            for i, group in enumerate(groups[ind]):
                output.extend(group)
                accumulated += sum([e.bondorder for e in group])/total_bo[ind]
                if accumulated > self.include_frac: break
        return output

    def make_atoms(self,material_id,trajname='final'):
        """HA HA HA"""
        pth = os.path.join(chargemol_folder,material_id,'%s.traj'%trajname)
        return ase.io.read(pth)

    def view_atoms(self,material_id,trajname='final'):
        from ase.visualize import view
        view(self.make_atoms(material_id,trajname))

    def plot_plotly_atoms(self, material_id, trajname = 'final'):
        import misc.plotly_atoms as mp
        graph = self.make_graph(material_id,trajname)
        return mp.PlotlyAtoms(graph).plot()

    def _get_sum_of_bond_order_data(self, material_id, trajname = 'final', show_indices = None):
        adj_matrix = self._make_adjacency_matrix(material_id, trajname)
        atoms_obj = self.make_atoms(material_id,trajname)
        sum_of_bond_orders = np.sum(adj_matrix, axis = 1)
        if show_indices is None:
            show_indices = range(adj_matrix.shape[0])

        indices =  [range(len(atoms_obj))[i] for i in show_indices]
        labels = ['%s-%s-%d'%(trajname,atom.symbol,atom.index) for atom in atoms_obj if atom.index in indices]
        return (labels, sum_of_bond_orders[indices])

    def plot_bond_order_analysis(self,material_id,trajname='final'
                                ,show_groups=True,filt=lambda x: True,show=True):
        import matplotlib; matplotlib.use('Qt4Agg')
        import matplotlib.pyplot as plt

        def get_symb(z):   return atoms[z].symbol

        edges    = self._make_edges(material_id,trajname) # {index:list of Edges from index}
        groups   = {ind: self._make_group(es) for ind,es in edges.items()}
        total_bo = {ind:sum([e.bondorder for e in es]) for ind,es in edges.items()}

        atoms = self.make_atoms(material_id, trajname)
        f,ax  = plt.subplots(nrows=1,ncols=1)
        plt.subplots_adjust(bottom=0.2)
        xmin,height   = 0, 0.2
        def align(h,v): # creates keyword dictionary for ax.text
            d = {'t':'top','b':'bottom','c':'center','l':'left','r':'right'}
            return {'horizontalalignment':d[h],'verticalalignment':d[v]}

        for ind in range(len(atoms)):
            if filt(atoms[ind]):
                xmax   = max([e.bondorder for e in edges[ind]])
                ax.hlines(ind,xmin,xmax)
                ax.vlines(xmin,ind-height,ind+height)
                ax.vlines(xmax,ind-height,ind+height)
                ax.text(0,ind,get_symb(ind)+' ',weight='bold',**align('r','c'))

                if show_groups and not self.jmol: # plot Group Information
                    accumulated = 0 # reset counter for bond strength accumulation
                    n_groups    = len(groups[ind])
                    for group_index,group in enumerate(groups[ind]):
                        over_thres  = accumulated > self.include_frac
                        weight,size = ('bold',10) if not over_thres else ('light',7)
                        lastgroup   = group_index == n_groups - 1
                        accumulated+= sum([e.bondorder for e in group])/total_bo[ind]
                        accum_txt   = '' if lastgroup else ' (%d%%)'%(accumulated*100)

                        mean = np.mean([e.bondorder for e in group])
                        ax.text(mean,ind+height, str(group_index) + accum_txt
                                ,size = size ,weight = weight, **align('c','b'))

                # make smaller groups for plotting atoms along the line
                # we have to treat each toNode index separately to not lose information
                elems = list(set([get_symb(edg.toNode) for edg in edges[ind]]))
                for elem in elems:
                    elem_edges = [edg for edg in edges[ind] if get_symb(edg.toNode) == elem]

                    smallgroups = self._make_group(elem_edges,0.03)

                    for smallgroup in smallgroups:
                        n = len(smallgroup) #multiplicity
                        s = np.mean([e.bondorder for e in smallgroup])
                        mult_txt = '(x%d)'%n if n > 1 else ''
                        ax.text(s,ind,elem,size=12,**align('c','c'))
                        ax.plot(s,ind,'ro',ms=15,mfc='r')
                        ax.text(s,ind-0.3,mult_txt,size=10,**align('c','t'))

        vtitle = 'jmol analysis with tol = %.3f'%self.jmol
        ctitle = ('Chargemol Bond analysis for %s: group_cut = '%trajname
                 +'%.2f'%(self.group_cut))
        ax.set_title(vtitle if self.jmol else ctitle)
        if show: plt.show()

    def make_graph(self,material_id,trajname='final',atoms=None):
        """Create NetworkX Graph Object"""
        if atoms is None: # ONLY useful to provide if doing jmol analysis
            atoms = self.make_atoms(material_id,trajname)

        adj_matrix          = self._make_adjacency_matrix(material_id,trajname)
        G                   = nx.MultiGraph( cell = atoms.get_cell()
                                            ,adj_matrix = adj_matrix)  # Initialize graph

        for i in range(len(atoms)): # add nodes
            G.add_node(i,symbol=atoms[i].symbol if self.colored else 1
                        ,position=atoms[i].position
                        ,index = i,magmom=atoms[i].magmom)

        edges = self._get_filtered_edges(material_id,trajname)
        for e in edges: e.add_edge_to_graph(G)
        return G
