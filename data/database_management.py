#External modules
import sqlite3, os, pickle, sql, json
from sql.operators import And
#Internal Modules
from CS230_Project.misc.sql_shortcuts import *
from CS230_Project.misc.utils import check_if_on_sherlock, sub_binds, replacer

################################################################################
"""
This module contains functions to query and update the input database, which
contains our dataset
"""
################################################################################

#Check for required enviromental variables
necessary_environ_variables = ['CS230_database_path','CS230_Project_Folder']
assert all([x in os.environ.keys() for x in necessary_environ_variables]),\
'Need all of the necessary environ variables to query the database'

#Set enviromental variable paths
project_folder          = os.environ['CS230_Project_Folder']
db_path                 = os.environ['CS230_database_path']
if check_if_on_sherlock():
    chargemol_folder    = os.environ['CS230_chargemol_folder']
else:
    chargemol_folder    =''

###########################
# Database Utilities
#--------------------------

class Query(object):
    """
    Read from a SQLite database
    """
    def __init__(self, cols=[PMG_Entries.pretty_formula], constraints=[], table = PMG_Entries
                , order = None,group = None,limit = None, deleted = False
                , db_path = db_path
                , verbose = False, distinct=False):

        self.constraints  = constraints
        self.cols         = cols
        self.table        = table
        self.order        = order
        self.group        = group
        self.limit        = limit
        self.deleted      = deleted
        self.db_path      = db_path
        self.verbose      = verbose

    def query(self, table = None, constraints=None, cols = None,row_factory = None,db_path=None):
        """ [STRING] -> CONSTRAINT -> [[sqloutput]] """
        if cols is None:        cols = self.cols
        if table is None:        table = self.table
        if constraints is None: constraints = self.constraints
        if row_factory is None: row_factory = lambda cursor, x: x
        if db_path is None: db_path = self.db_path

        command = self._command(constraints = constraints,cols = cols, table = table)
        if self.verbose == True: sub_binds(command)
        return sqlexecute(*command,db_path=db_path, row_factory=row_factory)

    def _command(self,constraints,cols, table):
        """Create sql command using python-sql"""
        if len(cols)==1 and '*' == str(cols[0]):
            cols = []
        self.last_command = table.select(*cols,where=And(constraints)
                      ,order_by=self.order,limit=self.limit,group_by=self.group)
        return self.last_command

    def query_dict(self, constraints = None, cols = None, dejson = False, deleted = None):
        """
        Returns a dictionary for every row returned (identical keys)
        """
        def dict_factory(cursor, row): return {col[0] : row[idx] for idx, col in enumerate(cursor.description)}
        def dict_factory_wdejson(cursor,row):
            d = {}
            for idx, col in enumerate(cursor.description):
                try:    d[col[0]] = json.loads(row[idx])
                except: d[col[0]] = row[idx]
            return d
        row_factory = dict_factory if dejson is False else dict_factory_wdejson
        return self.query(cols=cols, constraints = constraints, row_factory = row_factory)

    def query_col(self, col, constraints = None, table = None):
        """Query a single column in the database"""
        return self.query(cols = [col], constraints = constraints, table = table, row_factory = lambda cursor,x: x[0])

    def make_atoms(self, atoms_id_column):
        """returns a list of atoms corresponding to each row in the database"""
        output = self.query_col(PMG_Entries.atoms_obj)
        return map(traj_rebuild, output)


###########################
# Database Interfacing
#--------------------------
def sqlexecute(sqlcommand, binds = [],db_path=db_path,row_factory=None,hacks = []):
    assert sqlcommand.lower()[:6] in ['create','select','insert','delete','alter ','update','drop t'], "Sql Query weird "+sqlcommand
    connection = sqlite3.connect(db_path,timeout=30)
    if 'sqrt' in sqlcommand: connection.create_function('SQRT',1,math.sqrt)

    for h in hacks: sqlcommand = h(sqlcommand) # apply hacks to string

    cursor     = connection.cursor()
    cursor.row_factory = row_factory
    cursor.execute(sqlcommand, binds)
    output_array = cursor.fetchall()


    if sqlcommand.lower()[:6] != 'select': connection.commit()
    connection.close()
    return output_array

def sqlexecutemany(sqlcommand, binds = [],db_path=db_path):
    assert sqlcommand.lower()[:6] in ['create','select','insert','delete','alter ','update'], "Sql Query weird "+sqlcommand
    connection  = sqlite3.connect(db_path,timeout=60)
    cursor      = connection.cursor()
    cursor.executemany(sqlcommand, binds)
    if 'select' not in sqlcommand.lower(): connection.commit()
    cursor.close()
    connection.close()
    return


###########################
# Database Insertion/Updating
#--------------------------

def updateDB(set_column,id_column,ID,new_value,table,db_path=db_path):
    sqlexecute("UPDATE {0} SET {1}= ? WHERE {2} = ?".format(table,set_column,id_column),[new_value,ID],db_path=db_path)

def load_row_dictionary(row_dictionary,db_path=db_path):
    """
    INPUT: a dictionary for a single pymatgen entry for the PMG_Entries table
    """
    #List of required keys from PMG
    necessary_keys = ['pretty_formula'
                      ,'material_id'
                      ,'formation_energy_per_atom'
                      ,'e_above_hull'
                      ,'energy'
                      ,'energy_per_atom'
                      ,'num_atoms'
                      ,'cif'
                      ,'atoms_obj'
                      ,'chargemol'
                      ,'bonds_json']

    #Assert that all the necessary_keys are supplied
    assert all([x in row_dictionary.keys() for x in necessary_keys]), 'Need to provide all keys for PMG_Entries table'

    #Combine the keys and values into a string for insertion
    values = [row_dictionary[key] for key in necessary_keys]
    colNames,binds = ','.join(necessary_keys), values                           # Runtime column names and corresponding values
    command = 'INSERT into PMG_Entries (%s) VALUES (%s) '%( colNames, ','.join(['?']*len(binds)) ) # Make SQL command and
    sqlexecute(command,binds,db_path=db_path)

def update_chargemol():
    """As a preprocessing step, compute bond strengths for structures"""
    def read_bonds_json(chmol_folder):
        return json.load(open(os.path.join(chargemol_folder,chmol_folder,'bonds.json')))

    on_sherlock = check_if_on_sherlock()
    if on_sherlock:
        directories = os.listdir(chargemol_folder)
        already_updated = Query(constraints = [PMG_Entries.chargemol ==1]).query_col(PMG_Entries.material_id)
        fin_directories = filter(lambda x: os.path.exists(os.path.join(chargemol_folder,x,'bonds.json')) and x not in already_updated,directories)
        if len(fin_directories)>0:
            bonds_json_list = map(read_bonds_json, fin_directories)
            for bonds_json, material_id in zip(bonds_json_list, fin_directories):
                i = fin_directories.index(material_id)
                if i%100==0:
                    print 'Loaded {0} out of {1}'.format(i, len(bonds_json_list))
                dump_bonds = json.dumps(bonds_json)
                sqlexecute("UPDATE PMG_Entries SET chargemol= ?, bonds_json = ? WHERE material_id = ?",[1,dump_bonds,material_id],db_path=db_path)
            print 'Loaded a Total of {0} new bonds.json\'s'.format(len(bonds_json_list))
            return 1
        else:
            print 'No New Directories'
    else:
        print 'Can Only update DB on sherlock'
        return 0



###########################
# Database Creation
#--------------------------

def create_project_database(db_path = db_path):
    print 'creating DB at ',db_path
    fill_element(db_path)

    sqlexecute(("CREATE TABLE PMG_Entries   (id     integer primary key"
                                            +',pretty_formula               varchar not null'
                                            +',material_id                  varchar not null'
                                            +',formation_energy_per_atom    numeric not null'
                                            +',e_above_hull                 numeric'
                                            +',energy                       numeric'
                                            +',energy_per_atom              numeric'
                                            +',num_atoms                    integer not null'
                                            +',cif                          varchar not null'
                                            +',atoms_obj                    blob    not null'
                                            +',chargemol                    integer not null'
                                            +',bonds_json                   varchar'
                                            +',UNIQUE(material_id))'),db_path=db_path)

    os.system('chmod 777 '+db_path)
    return 1

def fill_element(db_path=db_path):
    from ase.data import atomic_masses,atomic_names,chemical_symbols,covalent_radii,reference_states
    from ase.build import bulk,molecule

    try:
        add_element(db_path)
    except sqlite3.OperationalError: pass

    # Constants
    # ---------
    sym_dict = {'fcc':225,'diamond':227,'bcc':229,'hcp':194 ,'sc':221
                ,'cubic':None ,'rhombohedral':None ,'tetragonal':None ,'monoclinic':None
                ,'orthorhombic':None,'diatom':'D*h','atom':'K*h'}

    ase_cols       = ['symbol','name','mass','radius','reference_phase','reference_spacegroup','reference_pointgroup']
    mendeleev_cols = ['atomic_number', 'atomic_weight', 'abundance_crust', 'abundance_sea', 'atomic_radius', 'atomic_volume',
                        'atomic_weight_uncertainty', 'boiling_point', 'covalent_radius_bragg', 'covalent_radius_cordero', 'covalent_radius_pyykko',
                        'covalent_radius_pyykko_double', 'covalent_radius_pyykko_triple', 'covalent_radius_slater', 'density', 'dipole_polarizability',
                        'econf', 'electron_affinity', 'en_allen', 'en_ghosh', 'en_pauling', 'evaporation_heat', 'fusion_heat', 'gas_basicity', 'geochemical_class',
                        'goldschmidt_class', 'group_id', 'heat_of_formation', 'is_radioactive', 'lattice_constant', 'lattice_structure', 'melting_point',
                        'metallic_radius', 'metallic_radius_c12', 'name', 'period', 'proton_affinity', 'symbol', 'thermal_conductivity', 'vdw_radius']
    cols = ase_cols+mendeleev_cols

    with open(os.environ['CS230_Project_Folder']+'/data/element.json','r') as f: eledicts=json.load(f)

    binds  = []
    qmarks = ','.join(['?']*len(cols))

    # Main Loop
    #----------
    for i in range(1,84):
        print i
        s,n,m,r = [x[i] for x in [chemical_symbols,atomic_names,atomic_masses,covalent_radii]]
        print m
        if reference_states[i] is None: rp,rsg,rpg=None,None,None
        elif 'a' in reference_states[i]:
            rp,rpg = 'bulk',None
            rsg = sym_dict[reference_states[i]['symmetry']]
            if i == 50: rsg = 227 #ase doesn't think tin is diamond
        else:
            rp,rsg = 'molecule',None
            rpg = sym_dict[reference_states[i]['symmetry']]
        ase_binds = [s,n,m,r,rp,rsg,rpg]

        m_binds = [eledicts[i-1].get(m_c) for m_c in mendeleev_cols]

        binds.append(ase_binds + m_binds)

    sqlexecutemany('INSERT INTO element (%s) VALUES (%s)'%(','.join(cols),qmarks),binds,db_path)


def add_element(db_path=db_path):
    sqlexecute(("CREATE TABLE element   (id     integer primary key" #ATOMIC NUMBER
                                        +',symbol  varchar not null'
                                        +',name    varchar not null'
                                        +',mass    numeric'
                                        +',radius  numeric not null'
                                        +',reference_phase varchar'
                                        +',reference_spacegroup integer '
                                        +',reference_pointgroup varchar'
                                        #brian's columns
                                        +',atomic_number                integer'
                                        +',atomic_weight                numeric'
                                        +',abundance_crust              numeric'
                                        +',abundance_sea                numeric'
                                        +',atomic_radius                numeric'
                                        +',atomic_volume                numeric'
                                        +',atomic_weight_uncertainty    numeric'
                                        +',boiling_point                numeric'
                                        +',covalent_radius_bragg        numeric'
                                        +',covalent_radius_cordero      numeric'
                                        +',covalent_radius_pyykko       numeric'
                                        +',covalent_radius_pyykko_double   numeric'
                                        +',covalent_radius_pyykko_triple   numeric'
                                        +',covalent_radius_slater   numeric'
                                        +',density                  numeric'
                                        +',dipole_polarizability    numeric'
                                        +',econf                    numeric'
                                        +',electron_affinity    numeric'
                                        +',en_allen             numeric'
                                        +',en_ghosh             numeric'
                                        +',en_pauling           numeric'
                                        +',evaporation_heat     numeric'
                                        +',fusion_heat          numeric'
                                        +',gas_basicity         numeric'
                                        +',geochemical_class    varchar'
                                        +',goldschmidt_class    varchar'
                                        +',group_id             integer'
                                        +',heat_of_formation    numeric'
                                        +',is_radioactive       bool'
                                        +',lattice_constant     numeric'
                                        +',lattice_structure    varchar'
                                        +',melting_point        numeric'
                                        +',metallic_radius      numeric'
                                        +',metallic_radius_c12  numeric'
                                        +',period               integer'
                                        +',proton_affinity      numeric'
                                        +',thermal_conductivity numeric'
                                        +',vdw_radius   numeric)'),db_path=db_path)
