# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:41:28 2025

@author: rohdo
"""

import scipy
import numpy as np
from rsjetstruct import RSjetStruct
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from plotTools import nFormatStr

dpi = 1000
N = 500
M = 10

kepsmin = 0
kepsmax = 2
kGammaImin = 0
kGammaImax = 1
kGammaIImin = 1
kGammaIImax = 3
gISMmin = 3/2
gISMmax = 7/2
gWindMin = 1/2
gWindMax = 3/2

KEPSls = np.linspace(kepsmin, kepsmax, M)
KGAMMAIls = np.linspace(kGammaImin, kGammaImax, M)
KGAMMAIIls = np.linspace(kGammaIImin, kGammaIImax, M)
GISMls = np.linspace(gISMmin, gISMmax, M)
GWINDls = np.linspace(gWindMin, gWindMax, M)

digits = 15

tobs = np.geomspace(1e-5, 1e20, N)
nu = 1
tcross = 1
Fnumaxrs_tcross = 1e-1
numrs_tcross_casea = 1e6
numrs_tcross_caseb = 1e6
numrs_tcross_casec = 1e8
nucutrs_tcross_casea = 1e8
nucutrs_tcross_caseb = 1e10
nucutrs_tcross_casec = 1e10
nuars_tcross_casea = 1e4
nuars_tcross_caseb = 1e8
nuars_tcross_casec = 1e12
keps = 1.1
kGamma_caseI = 0.5
kGamma_caseII = 1.5
p = 2.5

ISMcaseIa               =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 0)
ISMcaseIa_vary_keps     = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, KEPS, kGamma_caseI,  p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIa_vary_kGamma   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIls]
ISMcaseIa_vary_g        = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 0, g = G) for G in GISMls]

ISMcaseIb               =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 0)
ISMcaseIb_vary_keps     = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, KEPS, kGamma_caseI,  p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIb_vary_kGamma   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIls]
ISMcaseIb_vary_g        = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 0, g = G) for G in GISMls]

ISMcaseIc               =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 0)
ISMcaseIc_vary_keps     = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseI,  p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIc_vary_kGamma   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIls]
ISMcaseIc_vary_g        = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 0, g = G) for G in GISMls]

ISMcaseIIa              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 0)
ISMcaseIIa_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, KEPS, kGamma_caseII, p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIIa_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIIls]
ISMcaseIIa_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 0, g = G) for G in GISMls]

ISMcaseIIb              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 0)
ISMcaseIIb_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, KEPS, kGamma_caseII, p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIIb_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIIls]
ISMcaseIIb_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 0, g = G) for G in GISMls]

ISMcaseIIc              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 0)
ISMcaseIIc_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseII, p = p, k = 0)        for KEPS in KEPSls]
ISMcaseIIc_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, KGAMMA,        p = p, k = 0)        for KGAMMA in KGAMMAIIls]
ISMcaseIIc_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 0, g = G) for G in GISMls]

windCaseIa              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 2)
windCaseIa_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, KEPS, kGamma_caseI,  p = p, k = 2)        for KEPS in KEPSls]
windCaseIa_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIls]
windCaseIa_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 2, g = G) for G in GWINDls]

windCaseIb              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 2)
windCaseIb_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, KEPS, kGamma_caseI,  p = p, k = 2)        for KEPS in KEPSls]
windCaseIb_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIls]
windCaseIb_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 2, g = G) for G in GWINDls]

windCaseIc              =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 2)
windCaseIc_vary_keps    = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseI,  p = p, k = 2)        for KEPS in KEPSls]
windCaseIc_vary_kGamma  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIls]
windCaseIc_vary_g       = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 2, g = G) for G in GWINDls]

windCaseIIa             =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 2)
windCaseIIa_vary_keps   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, KEPS, kGamma_caseII, p = p, k = 2)        for KEPS in KEPSls]
windCaseIIa_vary_kGamma = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIIls]
windCaseIIa_vary_g      = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 2, g = G) for G in GWINDls]

windCaseIIb             =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 2)
windCaseIIb_vary_keps   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, KEPS, kGamma_caseII, p = p, k = 2)        for KEPS in KEPSls]
windCaseIIb_vary_kGamma = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIIls]
windCaseIIb_vary_g      = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 2, g = G) for G in GWINDls]

windCaseIIc             =  RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 2)
windCaseIIc_vary_keps   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseII, p = p, k = 2)        for KEPS in KEPSls]
windCaseIIc_vary_kGamma = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, KGAMMA,        p = p, k = 2)        for KGAMMA in KGAMMAIIls]
windCaseIIc_vary_g      = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 2, g = G) for G in GWINDls]

def plotFreqTemporalVary(rsjetstruct, filename, varystring, dpi = dpi):
    """"""
    
    if filename[19:21] == "II":
        kGammaBounds = (1, 3)
    else:
        kGammaBounds = (0, 1)
    
    varystringDict = {
        "keps"   : r"$k_\epsilon$",
        "kGamma" : r"$k_{\Gamma}$",
        "g"      : r"$g$"
    }
    
    varyamountDict = {
        "keps" : {
            0: r"$0<k_\epsilon<2$",
            2: r"$0<k_\epsilon<2$"
        },
        
        "kGamma" : {
            0: r"{}$<k_\Gamma<${}".format(*kGammaBounds),
            2: r"{}$<k_\Gamma<${}".format(*kGammaBounds)
        },
        
        "g" : {
            0: r"$3/2<g<7/2$",
            2: r"$1/2<g<3/2$"
        }  
    }
    
    paramstringDict = {
        "keps" : r" ($k_\Gamma = ${:.2f}, $g = ${:.2f})".format(rsjetstruct[0]._kGamma, rsjetstruct[0]._g),
        "kGamma" : r" ($k_\epsilon = ${:.2f}, $g = ${:.2f})".format(rsjetstruct[0]._keps, rsjetstruct[0]._g),
        "g" : r" ($k_\epsilon = ${:.2f}, $k_\Gamma = ${:.2f})".format(rsjetstruct[0]._keps, rsjetstruct[0]._kGamma)
    }
    
    tfrac = tobs/tcross
    plt.title(r"" + filename.lower().replace("plots/vary/", "").replace("case", " case ").replace("_", " ").replace(varystring.lower(), varyamountDict[varystring][rsjetstruct[0]._k]) + paramstringDict[varystring])
    plt.xscale("log")
    plt.xlabel(r"$t_{obs}/t_{\times}$")
    plt.yscale("log")
    plt.ylabel(r"$\nu$ or $F_{\nu}$")
    plt.plot(tfrac, rsjetstruct[0].Fnumaxrs(), color = "k", label = r"$F_{\nu}$",        alpha = 1)
    plt.plot(tfrac, rsjetstruct[0].numrs(),    color = "g", label = r"$\nu_{m}^{rs}$",   alpha = 1)
    plt.plot(tfrac, rsjetstruct[0].nucutrs(),  color = "b", label = r"$\nu_{cut}^{rs}$", alpha = 1)
    plt.plot(tfrac, rsjetstruct[0].nuars(),    color = "r", label = r"$\nu_{a}^{rs}$",   alpha = 1)
    
    for i in range(1, M):
        alpha = 1/(i + 1)
        
        plt.plot(tfrac, rsjetstruct[i].Fnumaxrs(), color = "k", alpha = alpha)
        plt.plot(tfrac, rsjetstruct[i].numrs(),    color = "g", alpha = alpha)
        plt.plot(tfrac, rsjetstruct[i].nucutrs(),  color = "b", alpha = alpha)
        plt.plot(tfrac, rsjetstruct[i].nuars(),    color = "r", alpha = alpha)
    
    plt.axvline(1, color = "k", linestyle = "--")
    plt.legend()
    
    plt.savefig(filename, dpi = dpi)
    plt.clf()
    
plotFreqTemporalVary(ISMcaseIa_vary_keps,   "plots/vary/ISMcaseIa_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIa_vary_kGamma, "plots/vary/ISMcaseIa_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIa_vary_g,      "plots/vary/ISMcaseIa_vary_g",      "g")

plotFreqTemporalVary(ISMcaseIb_vary_keps,   "plots/vary/ISMcaseIb_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIb_vary_kGamma, "plots/vary/ISMcaseIb_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIb_vary_g,      "plots/vary/ISMcaseIb_vary_g",      "g")

plotFreqTemporalVary(ISMcaseIc_vary_keps,   "plots/vary/ISMcaseIc_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIc_vary_kGamma, "plots/vary/ISMcaseIc_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIc_vary_g,      "plots/vary/ISMcaseIc_vary_g",      "g")

plotFreqTemporalVary(ISMcaseIIa_vary_keps,   "plots/vary/ISMcaseIIa_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIIa_vary_kGamma, "plots/vary/ISMcaseIIa_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIIa_vary_g,      "plots/vary/ISMcaseIIa_vary_g",      "g")

plotFreqTemporalVary(ISMcaseIIb_vary_keps,   "plots/vary/ISMcaseIIb_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIIb_vary_kGamma, "plots/vary/ISMcaseIIb_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIIb_vary_g,      "plots/vary/ISMcaseIIb_vary_g",      "g")

plotFreqTemporalVary(ISMcaseIIc_vary_keps,   "plots/vary/ISMcaseIIc_vary_keps",   "keps")
plotFreqTemporalVary(ISMcaseIIc_vary_kGamma, "plots/vary/ISMcaseIIc_vary_kGamma", "kGamma")
plotFreqTemporalVary(ISMcaseIIc_vary_g,      "plots/vary/ISMcaseIIc_vary_g",      "g")

plotFreqTemporalVary(windCaseIa_vary_keps,   "plots/vary/windCaseIa_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIa_vary_kGamma, "plots/vary/windCaseIa_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIa_vary_g,      "plots/vary/windCaseIa_vary_g",      "g")

plotFreqTemporalVary(windCaseIb_vary_keps,   "plots/vary/windCaseIb_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIb_vary_kGamma, "plots/vary/windCaseIb_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIb_vary_g,      "plots/vary/windCaseIb_vary_g",      "g")

plotFreqTemporalVary(windCaseIc_vary_keps,   "plots/vary/windCaseIc_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIc_vary_kGamma, "plots/vary/windCaseIc_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIc_vary_g,      "plots/vary/windCaseIc_vary_g",      "g")

plotFreqTemporalVary(windCaseIIa_vary_keps,   "plots/vary/windCaseIIa_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIIa_vary_kGamma, "plots/vary/windCaseIIa_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIIa_vary_g,      "plots/vary/windCaseIIa_vary_g",      "g")

plotFreqTemporalVary(windCaseIIb_vary_keps,   "plots/vary/windCaseIIb_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIIb_vary_kGamma, "plots/vary/windCaseIIb_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIIb_vary_g,      "plots/vary/windCaseIIb_vary_g",      "g")

plotFreqTemporalVary(windCaseIIc_vary_keps,   "plots/vary/windCaseIIc_vary_keps",   "keps")
plotFreqTemporalVary(windCaseIIc_vary_kGamma, "plots/vary/windCaseIIc_vary_kGamma", "kGamma")
plotFreqTemporalVary(windCaseIIc_vary_g,      "plots/vary/windCaseIIc_vary_g",      "g")