# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:08:43 2025

@author: rohdo
"""

import scipy
import numpy as np
from rsjetstruct import RSjetStruct
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from plotTools import alpha, nFormatStr

dpi = 500
N = 10000

digits = 15

tobs = np.geomspace(1e-5, 1e30, N)
nu = 1
tcross = 1
tjet = 1e10
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
keps = 0.1
kGamma_caseI = 0.5
kGamma_caseII = 1.5
p = 2.5

ISMcaseIa   = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 0, tjet = tjet)
ISMcaseIb   = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 0, tjet = tjet)
ISMcaseIc   = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 0, tjet = tjet)
ISMcaseIIa  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 0, tjet = tjet)
ISMcaseIIb  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 0, tjet = tjet)
ISMcaseIIc  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 0, tjet = tjet)

windCaseIa  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseI,  p = p, k = 2, tjet = tjet)
windCaseIb  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseI,  p = p, k = 2, tjet = tjet)
windCaseIc  = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseI,  p = p, k = 2, tjet = tjet)
windCaseIIa = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casea, nucutrs_tcross_casea, nuars_tcross_casea, keps, kGamma_caseII, p = p, k = 2, tjet = tjet)
windCaseIIb = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_caseb, nucutrs_tcross_caseb, nuars_tcross_caseb, keps, kGamma_caseII, p = p, k = 2, tjet = tjet)
windCaseIIc = RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, keps, kGamma_caseII, p = p, k = 2, tjet = tjet)

def plotFreqTemporal(rsjetstruct, filename, dpi = dpi):
    """"""
    tfrac = tobs/tcross
    # print(rsjetstruct.nuars()) # TODO debug print
    plt.title(filename.lower().replace("case", " case ").replace("plots/", "") +  r" " + "($p$={}, $k_\epsilon$={}, $k_\Gamma$={})".format(rsjetstruct._p, rsjetstruct._keps, rsjetstruct._kGamma))
    plt.xscale("log")
    plt.xlabel(r"$t_{obs}/t_{\times}$")
    plt.yscale("log")
    plt.ylabel(r"$\nu$ or $F_{\nu}$")
    alpha_Fnumaxrs = alpha(tfrac, rsjetstruct.Fnumaxrs()); plt.plot(tfrac, rsjetstruct.Fnumaxrs(), color = "k", label = r"$F_{\nu}:$"        + r"$\alpha$=" + nFormatStr(len(alpha_Fnumaxrs)).format(*alpha_Fnumaxrs))
    alpha_numrs = alpha(tfrac, rsjetstruct.numrs());       plt.plot(tfrac, rsjetstruct.numrs(),    color = "g", label = r"$\nu_{m}^{rs}:$"   + r"$\alpha$=" + nFormatStr(len(alpha_numrs)).format(*alpha_numrs))
    alpha_nucutrs = alpha(tfrac, rsjetstruct.nucutrs());   plt.plot(tfrac, rsjetstruct.nucutrs(),  color = "b", label = r"$\nu_{cut}^{rs}:$" + r"$\alpha$=" + nFormatStr(len(alpha_nucutrs)).format(*alpha_nucutrs))
    alpha_nuars = alpha(tfrac, rsjetstruct.nuars());       plt.plot(tfrac, rsjetstruct.nuars(),    color = "r", label = r"$\nu_{a}^{rs}:$"   + r"$\alpha$=" + nFormatStr(len(alpha_nuars)).format(*alpha_nuars))
    plt.axvline(1, color = "k", linestyle = "--")
    plt.axvline(rsjetstruct._tjet, color = "k", linestyle = ":", label = r"$t_{jet}$")
    plt.legend()
    
    plt.savefig(filename, dpi = dpi)
    plt.close()

def changerowlabels(df, rowlabels):
    """"""
    for i in range(len(rowlabels)):
        curr = df.index[i]
        df = df.rename(index = {curr: rowlabels[i]})
        
    return df

def table(rsjetstruct, filename):
    """"""
    df = pd.DataFrame.from_dict(rsjetstruct._alphaDict, orient='index').reset_index(drop=True)
    
    df = changerowlabels(df, list(rsjetstruct._alphaDict.keys()))
    df = df.transpose()
    
    with open(filename, "w") as f:
        f.write(df.to_string().replace("NaN", "---"))

plotFreqTemporal(ISMcaseIa,   "plots/ISMcaseIa",   dpi = dpi)
plotFreqTemporal(ISMcaseIb,   "plots/ISMcaseIb",   dpi = dpi)
plotFreqTemporal(ISMcaseIc,   "plots/ISMcaseIc",   dpi = dpi) 
plotFreqTemporal(ISMcaseIIa,  "plots/ISMcaseIIa",  dpi = dpi)
plotFreqTemporal(ISMcaseIIb,  "plots/ISMcaseIIb",  dpi = dpi)
plotFreqTemporal(ISMcaseIIc,  "plots/ISMcaseIIc",  dpi = dpi)

plotFreqTemporal(windCaseIa,  "plots/windCaseIa",  dpi = dpi)
plotFreqTemporal(windCaseIb,  "plots/windCaseIb",  dpi = dpi)
plotFreqTemporal(windCaseIc,  "plots/windCaseIc",  dpi = dpi)
plotFreqTemporal(windCaseIIa, "plots/windCaseIIa", dpi = dpi)
plotFreqTemporal(windCaseIIb, "plots/windCaseIIb", dpi = dpi)
plotFreqTemporal(windCaseIIc, "plots/windCaseIIc", dpi = dpi)

table(ISMcaseIa, "tables/caseI.txt")
table(ISMcaseIIa, "tables/caseII.txt")
