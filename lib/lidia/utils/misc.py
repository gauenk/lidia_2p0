
import numpy as np

def optional_pair(pydict,key,default,dtype):
    value = optional(pydict,key,default,dtype)
    if not(hasattr(value,"__getitem__")):
        value = np.array([value,value],dtype=dtype)
    return value

def optional(pydict,field,default):
    if pydict is None: return default
    if field in pydict: return pydict[field]
    else: return default

def optional_rm(pydict,field):
    if pydict is None:return
    if field in pydict: del pydict[field]
