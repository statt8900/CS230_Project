#External Modules
import sql
from sql import Table
from sql.functions import Function,Substring,Round,Random
from sql.operators import In, NotIn, And, Not
from sql           import As,Table,Join,Flavor,Literal

Flavor.set(Flavor(paramstyle='qmark')) # python-sql internal setting
################################################################################
"""This module contains shortcuts for using python-sql when querying the database
of input samples"""
################################################################################

PMG_Entries = Table('PMG_Entries')

def AND(*args):
    args = list(filter(None,args)) #remove Nones and empty lists
    if   len(args)==0: return None
    elif len(args)==1: return args[0]
    return And(args)

def OR(*args):
    args = list(filter(None,args)) #remove Nones and empty lists
    if len(args)==0: return None
    return Or(args)
