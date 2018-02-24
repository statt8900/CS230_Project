#External Modules
import sql
from sql import Table
from sql.functions import Function,Substring,Round,Random
from sql.operators import In, NotIn

################################################################################
"""This module contains shortcuts for using python-sql when querying the database
of input samples"""
################################################################################

PMG_Entries = Table('PMG_Entries')
