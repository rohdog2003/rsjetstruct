# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:51:31 2025

@author: rohdo
"""
from rsjetstruct import RSjetStruct
import numpy as np
import scipy
import matplotlib.pyplot as plt
from plotTools import plotrect
import sympy
from solveTemporalSymbolically import *

dpi = 750

dlnFpeakdlnt = -0.7
dlnNuPeakdlnt = -0.49
x0 = [0.5, 0.5]
npoints = 256
maxfev = 1000
factor = 1

def plotParamSpaceNumerically(caseStr, IorIIstr, dpi = dpi):
    """"""

    xmin = 0
    xmax = 2
    
    if IorIIstr == "II":
        ymin = 1
        ymax = 10
    else:
        ymin = 0
        ymax = 1

    if caseStr == "windCase":
        k = 2
    else:
        k = 0

    # considering wind case I
    def dlnFnumaxrsdlnt(keps, kGamma, p = 2.5, g = 1):
        """"""
        rs = RSjetStruct(1, 1, 1, 1, 2, 3, 1, keps, kGamma, p = p, g = g, k = k)
        return rs._alphaDict["Fnumaxrs"][caseStr + IorIIstr]
    
    def dlnNumrsdlnt(keps, kGamma, p = 2.5, g = 1):
        """"""
        rs = RSjetStruct(1, 1, 1, 1, 2, 3, 1, keps, kGamma, p = p, g = g, k = k)
        return rs._alphaDict["numrs"][caseStr + IorIIstr]
    
    def dlnNuarsdlnt(keps, kGamma, p = 2.5, g = 1):
        """"""
        rs = RSjetStruct(1, 1, 1, 1, 2, 3, 1, keps, kGamma, p = p, g = g, k = k)
        return rs._alphaDict["nuars"][caseStr + IorIIstr + "b"]
    
    def system1(x, p, g):
        """"""
        keps = x[0]
        kGamma = x[1]
        
        F = dlnFnumaxrsdlnt(keps, kGamma, p = p, g = g)
        num = dlnNumrsdlnt(keps, kGamma, p = p, g = g)
        
        return [F - dlnFpeakdlnt,\
                num - dlnNuPeakdlnt]
        
    def system2(x, p, g):
        """"""
        keps = x[0]
        kGamma = x[1]
        
        F = dlnFnumaxrsdlnt(keps, kGamma, p = p, g = g)
        num = dlnNumrsdlnt(keps, kGamma, p = p, g = g)
        nua = dlnNuarsdlnt(keps, kGamma, p = p, g = g)
        
        return [F - (p - 1)/2 * (nua - num) - dlnFpeakdlnt,\
                nua - dlnNuPeakdlnt]
        
    p = np.linspace(2, 3, int(np.sqrt(npoints)))
    
    if k==2:
        g = np.linspace(1/2, 3/2, int(np.sqrt(npoints)))
    else:
        g = np.linspace(3/2, 7/2, int(np.sqrt(npoints)))
    
    sol1x = []
    sol1y = []
    sol2x = []
    sol2y = []
    
    for P in p:
        for G in g:
            sol1 = scipy.optimize.fsolve(system1, x0, args = (P, G), maxfev = maxfev, factor = factor)
            sol1x.append(sol1[0])
            sol1y.append(sol1[1])
            #print(system1(sol1, P, G)) # FIXME not zero
            
            sol2 = scipy.optimize.fsolve(system2, x0, args = (P, G), maxfev = maxfev, factor = factor)
            sol2x.append(sol2[0])
            sol2y.append(sol2[1])
            #print(system2(sol2, P, G)) # FIXME not zero
            
    if k == 2:
        plt.title(caseStr[:-4] + " " + IorIIstr + r" varying $2<p<3$ and $\frac{1}{2}<g<\frac{3}{2}$")
    else:
        plt.title(caseStr[:-4] + " " + IorIIstr + r" varying $2<p<3$ and $\frac{3}{2}<g<\frac{7}{2}$")
    
    plt.xlabel(r"$k_\epsilon$")
    plt.xlim(0, 4)
    plt.ylabel(r"$k_\Gamma$")
    plt.ylim(0, 3)
    plt.axvline(xmin, color = "k", linestyle = "--")
    plt.axvline(xmax, color = "k", linestyle = "--")
    plt.axhline(ymin, color = "k", linestyle = "--")
    plt.axhline(ymax, color = "k", linestyle = "--")
    plotrect(xmin, xmax, ymin, ymax, color = "b", alpha = 0.3)
    plt.scatter(sol1x, sol1y, color = "g", label = r"$\nu_m$ peak")
    plt.scatter(sol2x, sol2y, color = "r", label = r"$\nu_a$ peak")
    plt.legend(loc = "upper right")
    
    plt.savefig("plots/error/" + caseStr + IorIIstr + "paramBOAT", dpi = dpi)
    plt.cla()

def plotParamSpaceSymbolically(caseStr, IorIIstr, dpi = dpi):
    """"""
    xmin = 0
    xmax = 2
    
    if IorIIstr == "II":
        ymin = 1
        ymax = 10
    else:
        ymin = 0
        ymax = 1
    
    fullCaseStr = caseStr + IorIIstr
    
    keps_numPeak, kGamma_numPeak = solveSymbolicallyNumPeak(fullCaseStr, dlnFpeakdlnt, dlnNuPeakdlnt)
    keps_nuaPeak, kGamma_nuaPeak = solveSymbolicallyNuaPeak(fullCaseStr, dlnFpeakdlnt, dlnNuPeakdlnt)
    
    keps_numPeak_lambdified = sympy.lambdify([p, g], keps_numPeak, "numpy")
    kGamma_numPeak_lambdified = sympy.lambdify([p, g], kGamma_numPeak, "numpy")
    
    keps_nuaPeak_lambdified = sympy.lambdify([p, g], keps_nuaPeak, "numpy")
    kGamma_nuaPeak_lambdified = sympy.lambdify([p, g], kGamma_nuaPeak, "numpy")
    
    p_numerical = np.linspace(2, 3, int(np.sqrt(npoints)))
    
    if caseStr == "ISMcase":
        g_numerical = np.linspace(3/2, 7/2, int(np.sqrt(npoints)))
    else:
        g_numerical = np.linspace(1/2, 3/2, int(np.sqrt(npoints)))
    
    if caseStr == "windCase":
        plt.title(caseStr[:-4] + " " + IorIIstr + r" varying $2<p<3$ and $\frac{1}{2}<g<\frac{3}{2}$")
    else:
        plt.title(caseStr[:-4] + " " + IorIIstr + r" varying $2<p<3$ and $\frac{3}{2}<g<\frac{7}{2}$")
    
    plt.xlabel(r"$k_\epsilon$")
    plt.xlim(0, 4)
    plt.ylabel(r"$k_\Gamma$")
    plt.ylim(0, 3)
    plt.axvline(xmin, color = "k", linestyle = "--")
    plt.axvline(xmax, color = "k", linestyle = "--")
    plt.axhline(ymin, color = "k", linestyle = "--")
    plt.axhline(ymax, color = "k", linestyle = "--")
    plotrect(xmin, xmax, ymin, ymax, color = "b", alpha = 0.3)
    
    plt.plot(keps_numPeak_lambdified(p_numerical[0], g_numerical), kGamma_numPeak_lambdified(p_numerical[0], g_numerical), color = "g", label = r"$\nu_m$ peak")
    plt.plot(keps_nuaPeak_lambdified(p_numerical[0], g_numerical), kGamma_nuaPeak_lambdified(p_numerical[0], g_numerical), color = "r", label = r"$\nu_a$ peak")
    
    for P_num in p_numerical[1:]:
        plt.plot(keps_numPeak_lambdified(P_num, g_numerical), kGamma_numPeak_lambdified(P_num, g_numerical), color = "g")
        plt.plot(keps_nuaPeak_lambdified(P_num, g_numerical), kGamma_nuaPeak_lambdified(P_num, g_numerical), color = "r")
      
    plt.legend()    
    
    plt.savefig("plots/error/" + fullCaseStr + "paramBOATsymbolic", dpi = dpi)
    plt.cla()

plotParamSpaceSymbolically("windCase", "I")
plotParamSpaceSymbolically("ISMcase", "I")
plotParamSpaceSymbolically("windCase", "II")
plotParamSpaceSymbolically("ISMcase", "II")

plotParamSpaceNumerically("windCase", "I")
plotParamSpaceNumerically("ISMcase", "I")
plotParamSpaceNumerically("windCase", "II")
plotParamSpaceNumerically("ISMcase", "II")
