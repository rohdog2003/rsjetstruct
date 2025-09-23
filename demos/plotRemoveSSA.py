# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 12:43:23 2025

@author: rohdo
"""

import numpy as np
import matplotlib.pyplot as plt
from rsjetstruct.gsspectshapes import Spectrum
from obsfluxmax import smallNum
from obsfluxmax import obsFluxMax

dpi = 500
SMALLNUM = 1e-50 # smallNum

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


#               Spectrum(nu,      Fnumax,         nuac,         nusa ,          num,          nuc , k = 0, p = 2.5)
ISMspec1noSSA = Spectrum(nu,      Fnumax,            0,     SMALLNUM , nu_2_5_X_8_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec2noSSA = Spectrum(nu,      Fnumax,            0,     SMALLNUM , nu_1_4_X_7_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec3noSSAc= Spectrum(nu,      Fnumax,            0,     SMALLNUM , nu_X_X_4_X_X,         5e11 , k = 0, p =   p)
ISMspec3noSSAf= Spectrum(nu,      Fnumax,   SMALLNUM/2,     SMALLNUM , nu_X_X_4_X_X,         1e10 , k = 0, p =   p)
ISMspec4noSSA = Spectrum(nu,      Fnumax,   SMALLNUM/2,     SMALLNUM , nu_3_3_X_9_X,         5e11 , k = 0, p =   p)
ISMspec5noSSA = Spectrum(nu,      Fnumax,   SMALLNUM/2,     SMALLNUM , nu_X_X_X_X_9, nu_X_X_X_X_11, k = 0, p =   p)

obsFluxMax1 = obsFluxMax(         Fnumax,            0, nu_1_4_X_7_X , nu_2_5_X_8_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec1      = Spectrum(nu, obsFluxMax1,            0, nu_1_4_X_7_X , nu_2_5_X_8_X, nu_3_3_X_9_X , k = 0, p =   p)
obsFluxMax2 = obsFluxMax(         Fnumax,            0, nu_2_5_X_8_X , nu_1_4_X_7_X, nu_3_3_X_9_X , k = 0, p =   p)
ISMspec2      = Spectrum(nu, obsFluxMax2,            0, nu_2_5_X_8_X , nu_1_4_X_7_X, nu_3_3_X_9_X , k = 0, p =   p)
obsFluxMax3c= obsFluxMax(         Fnumax,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,         5e11 , k = 0, p =   p)
ISMspec3c     = Spectrum(nu,obsFluxMax3c,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,         5e11 , k = 0, p =   p)
obsFluxMax3f= obsFluxMax(         Fnumax,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,         1e10 , k = 0, p =   p)
ISMspec3f     = Spectrum(nu,obsFluxMax3f,            0, nu_X_X_6_X_X , nu_X_X_4_X_X,         1e10 , k = 0, p =   p)
obsFluxMax4 = obsFluxMax(         Fnumax, nu_1_4_X_7_X, nu_2_5_X_8_X , nu_3_3_X_9_X,         5e11 , k = 0, p =   p)
ISMspec4      = Spectrum(nu, obsFluxMax4, nu_1_4_X_7_X, nu_2_5_X_8_X , nu_3_3_X_9_X,         5e11 , k = 0, p =   p)
obsFluxMax5 = obsFluxMax(         Fnumax, nu_X_X_X_X_7, nu_X_X_X_X_10, nu_X_X_X_X_9, nu_X_X_X_X_11, k = 0, p =   p)
ISMspec5      = Spectrum(nu, obsFluxMax5, nu_X_X_X_X_7, nu_X_X_X_X_10, nu_X_X_X_X_9, nu_X_X_X_X_11, k = 0, p =   p)

def plotRemoveSSA(filename, spec, specNoSSA, p = p, dpi = dpi):
    """"""
    nu, Fnu = spec.spectrum()
    nu_noSSA, Fnu_noSSA = specNoSSA.spectrum() # TODO for spec 4 and 5 ZeroDivisionError: float division by zero at File ~\OneDrive\Desktop\Python\rsjetstruct\spectrum.py:62
    
    plt.title(filename.lower().replace("spectrum", " spectrum ").replace("plots/removessa/", "") + " (p = {:.1f})".format(p))
    plt.xscale("log")
    plt.xlabel(r"$\nu$")
    plt.yscale("log")
    plt.ylabel(r"$F_\nu$")
    plt.plot(nu, Fnu, color = "k", label = "SSA")
    plt.plot(nu_noSSA, Fnu_noSSA, color = "b", linestyle = ":", label = "no SSA")
    plt.legend()
    
    plt.savefig(filename, dpi = dpi)
    plt.cla()
    
plotRemoveSSA("plots/removeSSA/spectrum1", ISMspec1, ISMspec1noSSA)
plotRemoveSSA("plots/removeSSA/spectrum2", ISMspec2, ISMspec2noSSA)
plotRemoveSSA("plots/removeSSA/spectrum3c", ISMspec3c, ISMspec3noSSAc)
plotRemoveSSA("plots/removeSSA/spectrum3f", ISMspec3f, ISMspec3noSSAf)
plotRemoveSSA("plots/removeSSA/spectrum4", ISMspec4, ISMspec4noSSA)
plotRemoveSSA("plots/removeSSA/spectrum5", ISMspec5, ISMspec5noSSA)
