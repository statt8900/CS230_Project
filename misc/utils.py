import os, pickle, sqlite3

def check_if_on_sherlock():
    hostname = os.environ['HOSTNAME'].lower()
    return 'sh'in hostname or  'gpu-' in hostname

def safeMkDir(pth):
    """Make directories even if they already exist"""
    try:            os.mkdir(pth)
    except OSError: print 'directory %s already exists ?!'%pth

def sub_binds(sql_select):
    """Prints a sql command in a human-readable way that can be copypasted
    into DB Browswer for SQLite."""

    keywords = ['INNER','FROM','HAVING','WHERE',"GROUP BY",", "]

    (sql_command,binds) = tuple(sql_select)

    for b in binds: sql_command=sql_command.replace('?',repr(b),1)

    replace_dict = {x:('\n\t'+x) for x in keywords}

    print '\n'+replacer(sql_command,replace_dict)+'\n'

def replacer(s,replace_dict):
    """Executes a series of string replacement operations, specified by a
    dictionary"""
    for k,v in replace_dict.items(): s = s.replace(k,v)
    return s

def traj_preparer(atoms):
	return sqlite3.Binary(pickle.dumps(atoms,pickle.HIGHEST_PROTOCOL))

def traj_rebuild(buffer_data):
	return pickle.loads(str(buffer_data))

def flatten(lol): return [item for sublist in lol for item in sublist] #flattens a List Of Lists to a list

negate   = lambda x: -x
