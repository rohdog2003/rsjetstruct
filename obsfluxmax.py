# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:41:42 2025

@author: rohdo
"""
import numpy as np

smallNum = 1e-50 # 1e-50 # a very small number close to zero

@np.vectorize
def obsFluxMax(Fnumax_nossa, nuac, nusa, num, nuc, p = 2.5, k = 0):
    """computes the observed maximum flux from the theoretical maximum if no
    synchrotron self absorption were to occur.
    """
    if nuac <= nusa <= num <= nuc: # spectrum 1
        return Fnumax_nossa
    elif nuac <= num <= nusa <= nuc: # spectrum 2
        return Fnumax_nossa * (nusa/num)**(-(p - 1)/2)
    elif nuac <= nusa and num <= nusa and nuc <= nusa: # spectrum 3
        if num <= nuc:
            return Fnumax_nossa * (nuc/num)**(-(p - 1)/2) * (nusa/nuc)**(-p/2)
        else:
            return Fnumax_nossa * (num/nuc)**(-1/2) * (nusa/num)**(-p/2)
    elif nuac <= nusa and nuc <= nusa and nusa <= num: # spectrum 4
        return Fnumax_nossa * (nusa/nuc)**(-1/2)
    elif nuac <= nusa <= nuc <= num: # spectrum 5 first case
        return Fnumax_nossa
    else:
        raise Exception("nuac must be smaller than nusa")
