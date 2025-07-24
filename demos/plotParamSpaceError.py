# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:46:05 2025

@author: rohdo
"""

from rsjetstruct import RSjetStruct
import numpy as np
import matplotlib.pyplot as plt

dpi = 500
N = 100

digits = 15

tobs = 1
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
keps = np.linspace(0, 2, N)
kGamma_caseI = np.linspace(0, 1, N)
kGamma_caseII = np.linspace(1, 3, N)
p = 2.5

ISMcaseIc   = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseI,  p = p, k = 0) for KEPS in keps]
ISMcaseIIc  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseII, p = p, k = 0) for KEPS in keps]
windCaseIc  = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseI,  p = p, k = 2) for KEPS in keps]
windCaseIIc = [RSjetStruct(tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross_casec, nucutrs_tcross_casec, nuars_tcross_casec, KEPS, kGamma_caseII, p = p, k = 2) for KEPS in keps]

def plotParamError(rsjetstruct, mediumStr, IorIIstr, toStr, keps, kGamma, bound1, bound2, dpi = dpi): # FIXME clear plot correctly
    """"""
    kepsSol = []
    kGammaSol = []
    
    for KEPSjetRS in rsjetstruct:
        if toStr == "ab":
            crossingNuStr = "numrs"
        else:
            crossingNuStr = "nucutrs"
                
        mask = (KEPSjetRS._alphaDict["nuars"][mediumStr + IorIIstr + toStr[1]] < KEPSjetRS._alphaDict[crossingNuStr][mediumStr + IorIIstr]) &\
               (KEPSjetRS._alphaDict["nuars"][mediumStr + IorIIstr + toStr[0]] > KEPSjetRS._alphaDict[crossingNuStr][mediumStr + IorIIstr])
            
        kGammaSolExtension = KEPSjetRS._kGamma[mask]
        kepsSolExtension = np.full(len(kGammaSolExtension), KEPSjetRS._keps)
        
        kepsSol.extend(kepsSolExtension)
        kGammaSol.extend(kGammaSolExtension)
    
    if toStr == "ab":
        condStr = r"$\frac{d\ln(\nu_{a,a}^{rs})}{d\ln(t)} > \frac{d\ln(\nu_{m}^{rs})}{d\ln(t)}$ and $\frac{d\ln(\nu_{a,b}^{rs})}{d\ln(t)} < \frac{d\ln(\nu_{m}^{rs})}{d\ln(t)}$"
    else:
        condStr = r"$\frac{d\ln(\nu_{a,b}^{rs})}{d\ln(t)} > \frac{d\ln(\nu_{cut}^{rs})}{d\ln(t)}$ and $\frac{d\ln(\nu_{a,c}^{rs})}{d\ln(t)} < \frac{d\ln(\nu_{cut}^{rs})}{d\ln(t)}$"
    
    plt.title(mediumStr.lower().replace("case", " case ") + IorIIstr + " (p={:.2f}, g={:.2f})".format(rsjetstruct[0]._p, rsjetstruct[0]._g))
    
    plt.xlim(keps[0], keps[-1])
    plt.xlabel(r"$k_\epsilon$")
    plt.ylim(kGamma[0], kGamma[-1])
    plt.ylabel(r"$k_\Gamma$")
    plt.scatter(kepsSol, kGammaSol, s = 40**2/N, label = condStr)
    plt.plot(keps, bound1(keps), color = "k", linestyle = "--")
    plt.plot(keps, bound2(keps), color = "k", linestyle = "--")
    plt.legend()
    
    plt.savefig("plots/error/" + mediumStr + IorIIstr + toStr[0].upper() + "to" + toStr[1].upper(), dpi = dpi)
    plt.clf()

def plotFastCoolError(rsjetstruct, mediumStr, IorIIstr, keps, kGamma, bound1, bound2, dpi = dpi):
    """"""
    kepsSol = []
    kGammaSol = []
    
    for KEPSjetRS in rsjetstruct:
        mask = KEPSjetRS._alphaDict["nucutrs"][mediumStr + IorIIstr] < KEPSjetRS._alphaDict["numrs"][mediumStr + IorIIstr]

        kGammaSolExtension = KEPSjetRS._kGamma[mask]
        kepsSolExtension = np.full(len(kGammaSolExtension), KEPSjetRS._keps)
        
        kepsSol.extend(kepsSolExtension)
        kGammaSol.extend(kGammaSolExtension)
        
    condStr = r"$\frac{d\ln(\nu_{cut}^{rs})}{d\ln(t)}<\frac{d\ln(\nu_m^{rs})}{d\ln(t)}$"
    
    plt.title(mediumStr.lower().replace("case", " case ") + IorIIstr + " (g={:.2f})".format(rsjetstruct[0]._g))
    
    plt.xlim(keps[0], keps[-1])
    plt.xlabel(r"$k_\epsilon$")
    plt.ylim(kGamma[0], kGamma[-1])
    plt.ylabel(r"$k_\Gamma$")
    plt.scatter(kepsSol, kGammaSol, s = 40**2/N, label = condStr)
    plt.plot(keps, bound1(keps), color = "k", linestyle = "--")
    plt.plot(keps, bound2(keps), color = "k", linestyle = "--")
    plt.legend()
    
    plt.savefig("plots/error/" + mediumStr + IorIIstr + "fast", dpi = dpi)
    plt.clf()

dummyBound = lambda KEPS : np.zeros_like(KEPS)

plotParamError(ISMcaseIc,   "ISMcase",  "I",  "ab", keps, kGamma_caseI,  dummyBound, dummyBound) # TODO bounds
plotParamError(ISMcaseIc,   "ISMcase",  "I",  "bc", keps, kGamma_caseI,  dummyBound, dummyBound) # TODO bounds
plotParamError(ISMcaseIIc,  "ISMcase",  "II", "ab", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds
plotParamError(ISMcaseIIc,  "ISMcase",  "II", "bc", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds
plotParamError(windCaseIc,  "windCase", "I",  "ab", keps, kGamma_caseI,   dummyBound, dummyBound) # TODO bounds
plotParamError(windCaseIc,  "windCase", "I",  "bc", keps, kGamma_caseI,  lambda KEPS : (1/3) * KEPS, lambda KEPS : 1/(3 * (p + 6)) * ((p + 4) * KEPS + 6))
plotParamError(windCaseIIc, "windCase", "II", "ab", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds
plotParamError(windCaseIIc, "windCase", "II", "bc", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds

plotFastCoolError(ISMcaseIc, "ISMcase", "I", keps, kGamma_caseI, dummyBound, dummyBound) # TODO bounds
plotFastCoolError(ISMcaseIIc, "ISMcase", "II", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds
plotFastCoolError(windCaseIc, "windCase", "I", keps, kGamma_caseI, dummyBound, lambda KEPS : (1/3) * KEPS) # TODO bounds
plotFastCoolError(windCaseIIc, "windCase", "II", keps, kGamma_caseII, dummyBound, dummyBound) # TODO bounds
