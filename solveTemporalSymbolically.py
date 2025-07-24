# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:12:29 2025

@author: rohdo
"""

from rsjetstruct import RSjetStruct
import sympy
import warnings

kG = sympy.Symbol("k")
a = sympy.Symbol("a")
A = a/kG
p = sympy.Symbol("p")
g = sympy.Symbol("g")

def symbolicAlphDict():
    """"""
    rsjetstruct = RSjetStruct(1, 1, 1, 1, 2, 3, 1, 0, 0)
    
    rsjetstruct._a = a
    rsjetstruct._A = A
    rsjetstruct._kGamma = kG
    rsjetstruct._p = p
    rsjetstruct._g = g
    
    return rsjetstruct._buildAlphaDict()

def solveSymbolicallyNumPeak(caseStr, dlnFpeak, dlnNuPeak):
    """"""
    alphaDict = symbolicAlphDict()
    
    dlnF = alphaDict["Fnumaxrs"][caseStr]
    dlnNum = alphaDict["numrs"][caseStr]
    
    system = [dlnF - dlnFpeak, dlnNum - dlnNuPeak]
    sol = sympy.solve(system, a, kG, dict = True)
    
    if _warning_nosolution(sol):
        return [0, 0]
    
    return list(sol[0].values())

def solveSymbolicallyNuaPeak(caseStr, dlnFpeak, dlnNuPeak):
    """"""
    alphaDict = symbolicAlphDict()
    
    dlnF = alphaDict["Fnumaxrs"][caseStr]
    dlnNum = alphaDict["numrs"][caseStr]
    dlnNua = alphaDict["nuars"][caseStr + "b"]
    
    system = [dlnF - (p - 1)/2 * (dlnNua - dlnNum) - dlnFpeak, dlnNua - dlnNuPeak]
    sol = sympy.solve(system, a, kG, dict = True)
    
    if _warning_nosolution(sol):
        return [0, 0]
    
    return list(sol[0].values())

def _warning_nosolution(sol):
    """"""
    if len(sol) == 0:
        warnings.warn("no solution found", RuntimeWarning)
        
        return True
    
    return False