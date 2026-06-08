# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 07:53:02 2026

@author: rohdo
"""

import numpy as np


def where(trutharray, funca, funcb, args = [], argsa = None, argsb = None): # TODO implement for nuars crossings
    """"""
    if argsa is None:
        argsa = args
        
    if argsb is None:
        argsb = args
    
    if isinstance(trutharray, np.ndarray):
        result = np.empty(trutharray.shape, dtype = float)
        falsearray = np.logical_not(trutharray)
        
        result[trutharray] = funca(*[arg[trutharray] for arg in argsa])
        result[falsearray] = funcb(*[arg[falsearray] for arg in argsb])
        
        return result
    
    else:
        if trutharray:
            return funca(*argsa)
        else:
            return funcb(*argsb)