# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:26:41 2025

@author: rohdo
"""

import numpy as np
import scipy
from inspect import signature

def multiDimCurveFit(fitFunc, X, Y, p0 = None, SIGMA = None, bounds = None):
    """Fits a multidimensional curve.
    
    Parameters
    ----------
    fitFunc : callable
        The model function, fitFunc(X, *args) with output shape (k, M) where X is shape (k, M).
    X : array_like
        The set of independent values taken with shape (k, M).
    Y : array_like
        The set of dependent values with shape (k, M).
    p0 : array_like, optional
        The initial guess for parameters.
    SIGMA : array_like
        The error in the dependent variabled with shape (k, M).
    bounds : 2-tuple of array_like
        Lower and upper parameter bounds.
    """
    
    def f(x, *args):
        """"""
        return flatten(fitFunc(X, *args))
        
    return scipy.optimize.curve_fit(f, flatten(X), flatten(Y), p0 = p0, sigma = flatten(SIGMA), bounds = bounds)
    
def chi2pdof(fitFunc, X, Y, SIGMA, args = []):
    """"""
    DOF = dof(fitFunc, X, Y)
    CHI2 = chi2(fitFunc, X, Y, SIGMA, args = args)
    
    return CHI2/DOF

def gof(fitFunc, X, Y, SIGMA, args = []):
    DOF = dof(fitFunc, X, Y)
    CHI2 = chi2(fitFunc, X, Y, SIGMA, args = args)
    
    #return scipy.special.gammainc(DOF/2, CHI2/2)
    return scipy.stats.chi2.cdf(CHI2, DOF)

def chi2(fitFunc, X, Y, SIGMA, args = []):
    """"""
    return np.sum((flatten(fitFunc(X, *args)) - flatten(Y))**2/flatten(SIGMA)**2)

def dof(fitFunc, X, Y):
    """"""
    return len(flatten(Y)) - numParams(fitFunc)

def numParams(fitFunc):
    """"""
    return len(signature(fitFunc).parameters) - 1

def flatten(Z):
    """"""
    return np.concatenate(np.array(Z, dtype = object))
