# -*- coding: utf-8 -*-
"""
Created on Fri May 30 09:31:03 2025

@author: rohdo
"""

import numpy as np
from rsjetstruct.gsspectshapes import Spectrum
import matplotlib.pyplot as plt

dpi = 500

nu = np.geomspace(1e8, 1e18, 200)
Fnumax = 10
p = 2.5

nu_1_4_X_7_X = 2e10
nu_2_5_X_8_X = 4e12
nu_3_3_X_9_X = 6e14
nu_X_X_4_X_X = 5e10
nu_X_X_6_X_X = 3e14
nu_X_X_X_X_7 = 1e10
nu_X_X_X_X_10 = 1e12
nu_X_X_X_X_11 = 1e14
nu_X_X_X_X_9 = 1e16


#           Spectrum(nu, Fnumax,         nuac,         nusa ,          num,          nuc , k = 0, p = 2.5)
ISMspec1  = Spectrum(nu, Fnumax,            0, nu_1_4_X_7_X , nu_2_5_X_8_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec2  = Spectrum(nu, Fnumax,            0, nu_2_5_X_8_X , nu_1_4_X_7_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec3  = Spectrum(nu, Fnumax,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,            0 , k = 0, p =   p)
ISMspec4  = Spectrum(nu, Fnumax, nu_1_4_X_7_X, nu_2_5_X_8_X , nu_3_3_X_9_X,            0 , k = 0, p =   p)
ISMspec5  = Spectrum(nu, Fnumax, nu_X_X_X_X_7, nu_X_X_X_X_10, nu_X_X_X_X_9, nu_X_X_X_X_11, k = 0, p =   p)

windSpec1 = Spectrum(nu, Fnumax,            0, nu_1_4_X_7_X , nu_2_5_X_8_X, nu_3_3_X_9_X , k = 2, p =   p)
windSpec2 = Spectrum(nu, Fnumax,            0, nu_2_5_X_8_X , nu_1_4_X_7_X, nu_3_3_X_9_X , k = 2, p =   p)
windSpec3 = Spectrum(nu, Fnumax,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,            0 , k = 2, p =   p)
windSpec4 = Spectrum(nu, Fnumax, nu_1_4_X_7_X, nu_2_5_X_8_X , nu_3_3_X_9_X,            0 , k = 2, p =   p)
windSpec5 = Spectrum(nu, Fnumax, nu_X_X_X_X_7, nu_X_X_X_X_10, nu_X_X_X_X_9, nu_X_X_X_X_11, k = 2, p =   p)

def plotFigure1(ISMspec, windSpec, filename, dpi = dpi):
    """"""
    plt.title(filename.lower().replace("spectrum", " spectrum ").replace("plots/", "") + " (p = {:.1f})".format(ISMspec._p))
    plt.xscale("log")
    plt.xlabel(r"$\nu$")
    plt.yscale("log")
    plt.ylim(top = 1e1 * np.max(ISMspec.spectrum()[1]), bottom = np.min(ISMspec.spectrum()[1]))
    plt.ylabel(r"$F_{\nu}$")
    plt.plot(*ISMspec.spectrum(), color = "black", label = "ISM")
    plt.plot(*windSpec.spectrum(), color = "m", linestyle = ":", label = "wind")
    
    #plt.plot(nu, ISMspec.spectrum()[1] * np.exp(-(nu/ISMspec._nuc - 1))**(nu > ISMspec._nuc), color = "black", alpha = 0.5)
    #plt.plot(nu, windSpec.spectrum()[1] * np.exp(-(nu/windSpec._nuc - 1))**(nu > windSpec._nuc), color = "m", linestyle = ":", alpha = 0.5)
    
    if ISMspec._nuac != 0:
        plt.axvline(ISMspec._nuac, color = "orange", linestyle = "--", label = r"$\nu_{ac}$")
    
    if ISMspec._nusa != 0:
        plt.axvline(ISMspec._nusa, color = "r", linestyle = "--", label = r"$\nu_{sa}$")
        
    if ISMspec._num != 0:
        plt.axvline(ISMspec._num, color = "g", linestyle = "--", label = r"$\nu_{m}$")
        
    if ISMspec._nuc != 0:
        plt.axvline(ISMspec._nuc, color = "b", linestyle = "--", label = r"$\nu_{c}$")
    
    plt.legend()
    plt.savefig(filename, dpi = dpi)
    plt.close()
    
plotFigure1(ISMspec1, windSpec1, "plots/spectrum1")
plotFigure1(ISMspec2, windSpec2, "plots/spectrum2")
plotFigure1(ISMspec3, windSpec3, "plots/spectrum3")
plotFigure1(ISMspec4, windSpec4, "plots/spectrum4")
plotFigure1(ISMspec5, windSpec5, "plots/spectrum5")