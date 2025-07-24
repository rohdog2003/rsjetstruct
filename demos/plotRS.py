# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 15:26:34 2025

@author: rohdo
"""
import numpy as np
import matplotlib.pyplot as plt
from rsjetstruct import RSjetStruct
import matplotlib as mpl

tcross          = 10**5.26083787e-01
Fnumaxrs_tcross = 10**1.00070158e+00
numrs_tcross    = 10**1.07027554e+01
nucutrs_tcross  = 10**3.81552983e+01 # 1e12
nuars_tcross    = 10**2.00000000e+00 
keps            = 2.41768484e-27 # 0.1
kGamma          = 1.26831080e+00
g               = 1.05661614e+00
k               = 2
p               = 2.531

def plotRS(k = k):
    """"""
    cmap = mpl.colormaps["hsv"]
    
    nu   = np.geomspace(1e7, 1e14, num = 200) 
    tobs = np.geomspace(1e-2, 1e2, num = 15)
    plt.xscale("log")
    plt.xlabel(r"$\nu$")
    plt.yscale("log")
    plt.ylabel(r"$F_\nu$")
    plt.ylim(bottom = 1e-4, top = 1e2)
    
    for i, t in enumerate(tobs):
        rsjetstruct = RSjetStruct(t, nu, tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, g = g, k = k, p = p)
        
        _, Fnu = rsjetstruct.spectrum()
        plt.plot(nu, Fnu, color = cmap(i/len(tobs)))
        
plotRS()
        
    