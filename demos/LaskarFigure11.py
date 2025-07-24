# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:41:57 2025

@author: rohdo
"""
import scipy.optimize as sciopt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from rsjetstruct import RSjetStruct
from fsjettophat import FSjetTopHat
from multidimcurvefit import multiDimCurveFit, chi2pdof, chi2, dof, gof
from plotTools import alpha, nFormatStr, plotErrorComparison, plotlogfan
import lcinterpolate
from lcinterpolate import LCinterpolate, argclosest

thr = 0.000001 # group equal elements
nobs = 1
nobs2 = 3 # number required to display in legend
k = 2
p = 2.531

dpi = 800

xmin = 3e7 # 3e8
xmax = 6e11
ymin = 2e-1
ymax = 2e1

LaskarTimes = np.array([3.46, 6.44, 12.44, 17.48, 28.33, 52.48, 76.42])
cmap = mpl.colormaps["gist_rainbow"]

t_cross_small = 1e-3
        
Fpeakobs = -0.7
Fpeakobserr = 0.02
nupeakobs = -0.49
nupeakobserr = 0.01

cbcolors = ["#553CA5", "#006EB2", "#96B4DF", "#F0E442", "#E69F00", "#D55E00", "#750000"]

def group(a,thr):
    """https://stackoverflow.com/questions/71677834/is-there-any-way-to-group-the-close-numbers-of-a-list-by-numpy"""
    x = np.sort(a)
    diff = x[1:]-x[:-1]
    gps = np.concatenate([[0],np.cumsum(diff>=thr)])
    return [x[gps==i] for i in range(gps[-1]+1)]

def find_nearest(a, a0):
    """https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    return a.flat[np.abs(a - a0).argmin()]

def getColorLegacy(tobs_grouped, i, t = None):
    """"""
    cmap = mpl.colormaps["gist_rainbow"] #["hsv"]
    MIN = np.min(tobs_grouped[0])
    MAX = np.max(tobs_grouped[-1])
    if t is None:
        VAL = np.mean(tobs_grouped[i])
    else:
        VAL = t
    
    #return cmap((VAL - MIN)/(MAX - MIN))
    return cmap(np.log(VAL - MIN + 1)/np.log(MAX - MIN + 1))

def getColor(tobs_grouped, i, t = None):
    """"""
    c1 = "#553CA5"
    c2 = "#006EB2"
    c3 = "#96B4DF"
    c4 = "#F0E442"
    c5 = "#E69F00"
    c6 = "#D55E00"
    c7 = "#750000"
    
    colors = np.array([c1, c2, c3, c4, c5, c6, c7])
    
    if t is None:
        j = argclosest(np.mean(tobs_grouped[i]), LaskarTimes)
    else:
        j = argclosest(t, LaskarTimes)
        
    return colors[j]

def plotLCfits(dpi = dpi):
    """"""
    tobs, nu, Fnu, err_Fnu, _, _, band, _, _ = np.loadtxt("apjlacbfadt1_mrt.txt", dtype = str, skiprows = 55).T
    
    nu      =      nu.astype(float)     # Hz
    tobs    =    tobs.astype(float)     # days
    Fnu     =     Fnu.astype(float)/1e3 # microjansky --> millijansky
    err_Fnu = err_Fnu.astype(float)/1e3 # microjansky --> millijansky
    
    lcinterpolate = LCinterpolate(band, nu, tobs, Fnu, err_Fnu)
    
    _, tobs_interp, F_interp, err_F_interp = lcinterpolate.lcinterpolate(LaskarTimes)
    
    t = np.geomspace(0.5, 1e2, 200)
    
    band_unique = np.unique(band)
    
    for b in band_unique:
        #plt.title(b)
        plt.xscale("log")
        plt.xlabel(r"$t$[days]")
        plt.yscale("log")
        plt.ylabel(r"$F_\nu$[mJy]")
        
        tobsB    =    tobs[band == b]
        FnuB     =     Fnu[band == b]
        err_FnuB = err_Fnu[band == b]
        
        tobs_interpB = tobs_interp[band == b]
        F_interpB = F_interp[band == b]
        err_F_interpB = err_F_interp[band == b]
        
        #if not(b in ["0.9mm", "1.1mm", "Ka", "Q", "U4"]): # TODO no fits for these
        plt.plot(t, lcinterpolate.lcfit(t, b))
        
        plotline, _, _ = plt.errorbar(tobsB, FnuB, yerr = err_FnuB, fmt = "o", label = "raw")
        plotline.set_markerfacecolor('none')
        
        lcinterp, goodness = lcinterpolate.lcfit(LaskarTimes, b, return_gof = True)
        
        for time in LaskarTimes:
            plt.axvline(time, color = "k", linestyle = "--", alpha = 0.5)
        
        plotline, _, _ = plt.errorbar(tobs_interpB, F_interpB, yerr = err_F_interpB, fmt = "d", label = "interpolated", alpha = 0.7)
        plotline.set_markerfacecolor('none')
        
        plt.title(b + ": p={:.2e}".format(goodness))
        
        plt.legend()
        plt.savefig("plots/fits/lcFits/" + b.replace(".",","), dpi = dpi)
        plt.cla()
    

def plotData(thr = thr, nobs = nobs, dpi = dpi):
    """"""
    tobs, nu, Fnu, err_Fnu, _, _, band, _, _ = np.loadtxt("apjlacbfadt1_mrt.txt", dtype = str, skiprows = 55).T

    nu      =      nu.astype(float)     # Hz
    tobs    =    tobs.astype(float)     # days
    Fnu     =     Fnu.astype(float)/1e3 # microjansky --> millijansky
    err_Fnu = err_Fnu.astype(float)/1e3 # microjansky --> millijansky
    
    lcinterpolate = LCinterpolate(band, nu, tobs, Fnu, err_Fnu)
    
    _, t_interp, F_interp, err_F_interp = lcinterpolate.lcinterpolate(LaskarTimes)
    
    tobs_grouped = group(tobs, thr)
    
    tobs_grouped_new = []
    nu_grouped = []
    Fnu_grouped =[]
    err_Fnu_grouped = []
    colors_grouped = []
    
    for i, TOBS_GROUP in enumerate(tobs_grouped):
        if len(TOBS_GROUP) < nobs:
            continue
        
        tobs_grouped_new.append(TOBS_GROUP)
        
        mask = tobs == TOBS_GROUP[0]
        
        for TOBS_GROUPELEMENT in TOBS_GROUP[1:]:
            mask = mask | (tobs == TOBS_GROUPELEMENT)
        
        nu_tobs      =      nu[mask];
        Fnu_tobs     =     Fnu[mask];
        err_Fnu_tobs = err_Fnu[mask];
        
        nu_grouped.append(nu_tobs)
        Fnu_grouped.append(Fnu_tobs)
        err_Fnu_grouped.append(err_Fnu_tobs)
        colors_grouped.append(getColor(tobs_grouped, i))
    
    for time in LaskarTimes:
        
        nu_time = nu[t_interp == time]
        Fnu_time = F_interp[t_interp == time]
        err_Fnu_time = err_F_interp[t_interp == time]
        
        plt.xlim(xmin, xmax)
        plt.xscale("log")
        plt.xlabel("Frequency [Hz]")
        plt.ylim(ymin, ymax)
        plt.yscale("log")
        plt.ylabel("Flux [mJy]")
        plotline, _, _ = plt.errorbar(nu_time, Fnu_time, yerr = err_Fnu_time,\
                                      fmt = "o", color = getColor(tobs_grouped, i, t = time))
        plotline.set_markerfacecolor('none')
        plotline.set_markeredgewidth(1)
        
    def norm(vmin = None, vmax = None):
        """"""
        return lambda t : getColor([[vmin], [vmax]], 0, t = t)
        

    plt.savefig("plots/fits/data", dpi = dpi)
    
    return tobs_grouped_new, nu_grouped, Fnu_grouped, err_Fnu_grouped, colors_grouped

def plotBreakFreq(fileName, tobs_grouped, tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, p = p, k = k, g = 1, dpi = dpi):
    """"""
    tobs = np.geomspace(tcross * 1e-1, np.max(tobs_grouped[-1]), 1000)
    
    rsjetstruct = RSjetStruct(tobs, 1, tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, p = p, k = k, g = g)
    
    mPL = alpha(tobs, rsjetstruct.numrs())
    cutPL = alpha(tobs, rsjetstruct.nucutrs())
    aPL = alpha(tobs, rsjetstruct.nuars())
    FPL = alpha(tobs, rsjetstruct.Fnumaxrs())
    
    plt.xscale("log")
    plt.xlabel(r"$t_{obs} [d]$")
    plt.yscale("log")
    plt.ylabel(r"$F_{\nu,max}^{rs}$ [mJy] or $\nu_b^{rs}$ [GHz]")
    plt.axvline(tcross, linestyle = "--", color = "k", label = r"$t_\times$")
    
    if nucutrs_tcross < 1e15:
        plt.plot(tobs, rsjetstruct.nucutrs()/1e9, color = "#006EB2", label = r"$\nu_{cut}^{rs}:$" + nFormatStr(len(cutPL)).format(*cutPL))
        
    plt.plot(tobs, rsjetstruct.numrs()/1e9, color = "#553CA5", label = r"$\nu_m^{rs}$:" + nFormatStr(len(mPL)).format(*mPL))
    plt.plot(tobs, rsjetstruct.nuars()/1e9, color = "#750000", label = r"$\nu_a^{rs}$:" + nFormatStr(len(aPL)).format(*aPL))
    plt.plot(tobs, rsjetstruct.Fnumaxrs(), color = "k", label = r"$F_{\nu,max}^{rs}:$" + nFormatStr(len(FPL)).format(*FPL))
    plt.legend()
    
    plt.savefig(fileName + "_breakFreq", dpi = dpi)
    plt.cla()

def plotFit(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, k = k, p = p, dpi = dpi):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logtcross, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, keps, kGamma, g):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = 10**logtcross
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, k = k, p = p, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    #              logtcross, logFnumax,  lognumrs, lognucutrs,  lognuars, keps, kGamma,   g, NOTE: logarithms are in base 10
    lowerBounds = [       -3,        -1,         5,         12,         5,    0,      0, 1/2]
    upperBounds = [        2,         4,        20,         20,        20,    2,     10,   5]
    p0 =          [      0.5,       1.1,        10,         12,         8,  0.1,      0,   1]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:-3], *popt[-3:])
    
    plt.title(r"$\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, *10**popt[:-3], popt[-3], popt[-2], k = k, p = p, g = popt[-1])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fit", dpi = dpi)
    plt.close()
    
    plotBreakFreq("plots/fits/fit", tobs_grouped, *10**popt[:5], popt[5], popt[6], p = p, k = k, g = popt[7])
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    return popt

def plotFitpRS(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, p = p, k = k, dpi = 1000):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, keps, kGamma, g, pRS):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = t_cross_small
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    # g = 1
    # tdec = 0.001
    # tsed = 1 # 3.45 # TODO match Laskar et al. 2023
    
    # nua0  = 2.3e9*(tdec/tsed)**(-(33.*g+36.)/(70.*g+35.)); print("{:.2e}".format(nua0))
    # num0  = 1.2e11*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(num0))
    # nuc0  = 1e15*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(nuc0))
    # fnum0 = 94*(tdec/tsed)**(-(11.*g+12.)/(14.*g+7.)); print(fnum0, "\n")
    
    #              logFnumax,  lognumrs, lognucutrs,  lognuars, keps, kGamma,   g, pRS NOTE: logarithms are in base 10
    lowerBounds = [        3,        10,         14,        10,    0,      0, 1/2,   2]
    upperBounds = [        9,        20,         30,        20,    2,      3, 3/2,   3]
    p0 =          [     3.98,     13.77,         25,     12.57,    0,    0.1,   1, 2.3]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:-4], *popt[-4:])
    
    plt.title(r"Structured Jet Model: $\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    oneDayFit = RSjetStruct(1, nu, t_cross_small, *10**popt[:-4], popt[-4], popt[-3], k = k, p = popt[-1], g = popt[-2])
    
    err_logFnumax, err_lognumrs, err_lognucutrs, err_lognuars, err_keps, err_kGamma, err_g, err_pRS = np.sqrt(np.diag(pcov))
    err_A = np.sqrt((err_keps/oneDayFit._kGamma)**2 + (oneDayFit._keps/oneDayFit._kGamma**2 * err_kGamma)**2)
    
    print("numrs(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.numrs(),     np.log(10) * oneDayFit.numrs() * err_lognumrs))
    print("nuars(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.nuars(),     np.log(10) * oneDayFit.nuars() * err_lognuars))
    print("nucutrs(1 d) : {:.2e}±{:.2e}".format( oneDayFit.nucutrs(), np.log(10) * oneDayFit.nucutrs() * err_lognucutrs))
    print("Fnumaxrs(1 d): {:.2e}±{:.2e}".format(oneDayFit.Fnumaxrs(), np.log(10) * oneDayFit.Fnumaxrs() * err_logFnumax))
    print("A            : {:.2e}±{:.2e}".format(        oneDayFit._A,                                             err_A))
    print("pRS          : {:.2f}±{:.2e}".format(        oneDayFit._p,                                           err_pRS))
    print()
    print("keps         : {:.2f}±{:.2e}".format(  oneDayFit._keps,   err_keps))
    print("kGamma       : {:.2f}±{:.2e}".format(oneDayFit._kGamma, err_kGamma))
    print()
    print("Fpeak        : ∝t^({:.2f}±{:.2e})".format(oneDayFit._alphaDict["Fnumaxrs"]["windCaseII"], 3/(4 - oneDayFit._A) * err_A)) # assuming peak at num wind II
    print("nupeak       : ∝t^({:.2f}±{:.2e})".format(   oneDayFit._alphaDict["numrs"]["windCaseII"],                            0))
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:-4], popt[-4], popt[-3], k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fitpRS", dpi = dpi)
    plt.savefig("plots/fits/fitpRS.svg")
    plt.close()
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    plotBreakFreq("plots/fits/fitpRS", tobs_grouped, t_cross_small, *10**popt[:4], popt[4], popt[5], p = popt[7], k = k, g = popt[6])
    
    return popt

def plotFitpRSoneLessFree(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, p = p, k = k, dpi = 1000):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    kGamma = 1.01
    
    def fitFunc(NU, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, A, g, pRS):
        """"""
        keps = A * kGamma
        
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = t_cross_small
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    # g = 1
    # tdec = 0.001
    # tsed = 1 # 3.45 # TODO match Laskar et al. 2023
    
    # nua0  = 2.3e9*(tdec/tsed)**(-(33.*g+36.)/(70.*g+35.)); print("{:.2e}".format(nua0))
    # num0  = 1.2e11*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(num0))
    # nuc0  = 1e15*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(nuc0))
    # fnum0 = 94*(tdec/tsed)**(-(11.*g+12.)/(14.*g+7.)); print(fnum0, "\n")
    
    #              logFnumax,  lognumrs, lognucutrs,  lognuars,   A,   g, pRS NOTE: logarithms are in base 10
    lowerBounds = [        3,        10,         14,        10,   0, 1/2,   2]
    upperBounds = [        9,        20,         30,        20, 1.5, 3/2,   3]
    p0 =          [     3.98,     13.77,         25,     12.57,   0,   1, 2.3]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:-3], *popt[-3:])
    
    plt.title(r"Structured Jet Model: $\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    oneDayFit = RSjetStruct(1, nu, t_cross_small, *10**popt[:-3], popt[-3] * kGamma, kGamma, k = k, p = popt[-1], g = popt[-2])
    
    err_logFnumax, err_lognumrs, err_lognucutrs, err_lognuars, err_A, err_g, err_pRS = np.sqrt(np.diag(pcov))
    
    print("numrs(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.numrs(),     np.log(10) * oneDayFit.numrs() * err_lognumrs))
    print("nuars(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.nuars(),     np.log(10) * oneDayFit.nuars() * err_lognuars))
    print("nucutrs(1 d) : {:.2e}±{:.2e}".format( oneDayFit.nucutrs(), np.log(10) * oneDayFit.nucutrs() * err_lognucutrs))
    print("Fnumaxrs(1 d): {:.2e}±{:.2e}".format(oneDayFit.Fnumaxrs(), np.log(10) * oneDayFit.Fnumaxrs() * err_logFnumax))
    print("A            : {:.2e}±{:.2e}".format(        oneDayFit._A,                                             err_A))
    print("pRS          : {:.2f}±{:.2e}".format(        oneDayFit._p,                                           err_pRS))
    print()
    
    
    Fpeak = oneDayFit._alphaDict["Fnumaxrs"]["windCaseII"]
    Fpeakerr = 3/(4 - oneDayFit._A)**2 * err_A
    nupeak = oneDayFit._alphaDict["numrs"]["windCaseII"]
    nupeakerr = 0
    print("Fpeak        : ∝t^({:.2f}±{:.2e})".format(Fpeak, Fpeakerr)) # assuming peak at num wind II
    print("nupeak       : ∝t^({:.2f}±{:.2e})".format(nupeak, nupeakerr))
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:-3], popt[-3] * kGamma, kGamma, k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        #plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        #plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fitpRSoneLessFreeLessDetail", dpi = dpi)
    plt.savefig("plots/fits/fitpRSoneLessFreeLessDetail.svg")
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:-3], popt[-3] * kGamma, kGamma, k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        #plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fitpRSoneLessFree", dpi = dpi)
    plt.savefig("plots/fits/fitpRSoneLessFree.svg")
    plt.close()
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    plotBreakFreq("plots/fits/fitpRSoneLessFree", tobs_grouped, t_cross_small, *10**popt[:4], popt[4] * kGamma, kGamma, p = popt[-1], k = k, g = popt[-2])
    
    return popt

def plotTopHatFitpRS(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, p = p, k = k, dpi = 1000):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, g, pRS):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = t_cross_small
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, 0, 0, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    # g = 1
    # tdec = 0.001
    # tsed = 1 # 3.45 # TODO match Laskar et al. 2023
    
    # nua0  = 2.3e9*(tdec/tsed)**(-(33.*g+36.)/(70.*g+35.)); print("{:.2e}".format(nua0))
    # num0  = 1.2e11*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(num0))
    # nuc0  = 1e15*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(nuc0))
    # fnum0 = 94*(tdec/tsed)**(-(11.*g+12.)/(14.*g+7.)); print(fnum0, "\n")
    
    #              logFnumax,  lognumrs, lognucutrs,  lognuars,   g, pRS NOTE: logarithms are in base 10
    lowerBounds = [        3,        10,         14,        10, 1/2,   2]
    upperBounds = [        9,        20,         30,        20, 3/2,   3]
    p0 =          [     3.98,     13.77,         25,     12.57,   1, 2.3]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:4], *popt[4:])
    
    plt.title(r"Top Hat Jet Model: $\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    oneDayFit = RSjetStruct(1, nu, t_cross_small, *10**popt[:4], 0, 0, k = k, p = popt[-1], g = popt[-2])
    
    err_logFnumax, err_lognumrs, err_lognucutrs, err_lognuars, err_g, err_pRS = np.sqrt(np.diag(pcov))
    
    print("numrs(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.numrs(),                                                    np.log(10) * oneDayFit.numrs() * err_lognumrs))
    print("nuars(1 d)   : {:.2e}±{:.2e}".format(   oneDayFit.nuars(),                                                    np.log(10) * oneDayFit.nuars() * err_lognuars))
    print("nucutrs(1 d) : {:.2e}±{:.2e}".format( oneDayFit.nucutrs(),                                                np.log(10) * oneDayFit.nucutrs() * err_lognucutrs))
    print("Fnumaxrs(1 d): {:.2e}±{:.2e}".format(oneDayFit.Fnumaxrs(),                                                np.log(10) * oneDayFit.Fnumaxrs() * err_logFnumax))
    print("pRS          : {:.2f}±{:.2e}".format(        oneDayFit._p,                                                                                          err_pRS))
    print("g            : {:.2f}±{:.2e}".format(        oneDayFit._g,                                                                                            err_g))
    print()
    print("Fpeak        : {:.2f}±{:.2e}".format(oneDayFit._alphaDict["Fnumaxrs"]["windCaseI"], 13/(7 * (1 + 2 * oneDayFit._g)**2) * err_g))
    print("nupeak       : {:.2f}±{:.2e}".format(   oneDayFit._alphaDict["numrs"]["windCaseI"], 33/(7 * (1 + 2 * oneDayFit._g)**2) * err_g))
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:4], 0, 0, k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/topHatFitpRSlessDetail", dpi = dpi)
    plt.savefig("plots/fits/topHatFitpRSlessDetail.svg")
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:4], 0, 0, k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        #plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/topHatFitpRS", dpi = dpi)
    plt.savefig("plots/fits/topHatFitpRS.svg")
    plt.close()
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    plotBreakFreq("plots/fits/topHatFitpRS", tobs_grouped, t_cross_small, *10**popt[:4], 0, 0, p = popt[-1], k = k, g = popt[-2])
    
    return popt

def plotFitpRSplusGG23(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, p = p, k = k, dpi = dpi):
    """"""
    FSparams = (1e-2, 1e-4, 1, 0.33, 2e55/1e52) # FIXME parameters aren't reproducing FS curves of 
    z = 0.151
    pFS = 2.4
    xie = 0.01
    
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logtcross, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, keps, kGamma, g, pRS):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], *FSparams, tdays, z, p = pFS, xie = xie, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = 10**logtcross
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    #              logtcross, logFnumax,  lognumrs, lognucutrs,  lognuars,  keps, kGamma,   g, pRS NOTE: logarithms are in base 10
    lowerBounds = [       -3,        -1,         5,         12,         5,     0,      0, 1/2,   2]
    upperBounds = [        2,         4,        20,         20,        20,     2,     10,   5,   3]
    p0 =          [      0.5,       1.1,        10,         13,         8,     0,      2,   1, 2.3]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:-4], *popt[-4:])
    
    plt.title(r"$\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, *10**popt[:-4], popt[-4], popt[-3], k = k, p = popt[-1], g = popt[-2])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, *FSparams, t, z, p = pFS, xie = xie, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fitpRSplusGG23", dpi = dpi)
    plt.close()
    
    plotBreakFreq("plots/fits/fitpRSplusGG23", tobs_grouped, *10**popt[:5], popt[5], popt[6], p = popt[8], k = k, g = popt[7])
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    return popt

def plotFitpRSplusFreeFS(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, p = p, k = k, dpi = dpi):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logtcross, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, keps, kGamma, g, pRS, logepse, logepsB, logAstar, logE52, pFS):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**logepse, 10**logepsB, 1, 10**logAstar, 10**logE52, tdays, 0.151, p = pFS, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = 10**logtcross
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    #              logtcross, logFnumax,  lognumrs, lognucutrs,  lognuars, keps, kGamma,   g, pRS,      logepse,      logepsB,     logAstar,   logE52, pFS, NOTE: logarithms are in base 10
    lowerBounds = [       -3,        -1,         5,         12,         5,    0,      0, 1/2,   2,           -2,           -4,           -5,    53/52,   2]
    upperBounds = [        1,         4,        20,         20,        20,    2,     10, 3/2,   3,         -0.1,         -0.1,           -2,    56/52,   3]
    p0 =          [      0.5,       1.1,        10,         13,      10.1,  0.1,      0,   1, 2.3,       -0.616,       -0.221,       -3.612, 54.04/52,   p]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:5], *popt[5:9], *10**popt[9:12], 10**popt[12] * 1e52, popt[13]) 
    
    plt.title(r"$\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, *10**popt[:5], popt[5], popt[6], k = k, p = popt[8], g = popt[7]) # TODO
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**popt[9], 10**popt[10], 1, 10**popt[11], 10**popt[12], t, 0.151, p = popt[13], k = k) # TODO
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/fitpRSplusFreeFS", dpi = dpi)
    plt.close()
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    
    plotBreakFreq("plots/fits/fitpRSplusFreeFS", tobs_grouped, *10**popt[:5], popt[5], popt[6], p = popt[8], k = k, g = popt[7])
    
    return popt

def plotLaskarFit(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, dpi = dpi):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    pRS = 2.05 # 2.05
    
    def fitFunc(NU, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, g):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = 10**np.log10(t_cross_small)
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, 0, 0, k = k, p = pRS, g = g)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    # #              logtcross, logFnumax,  lognumrs, lognucutrs,  lognuars,   g, NOTE: logarithms are in base 10
    # lowerBounds = [       -3,        -1,         5,         12,         5, 1/2]
    # upperBounds = [        2,         4,        20,         20,        20,   5]
    # p0 =          [      0.5,       1.1,        10,         12,         8,   1]
    # bounds =      [lowerBounds, upperBounds]
    # popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    # print(*10**popt[:-1], popt[-1])
    
    g = 5
    tdec = 0.001
    tsed = 1 # 3.45 # TODO match Laskar et al. 2023
    
    # nua0  = 2.3e9*(tdec/tsed)**(-(33.*g+36.)/(70.*g+35.)); print("{:.2e}".format(nua0))
    # num0  = 1.2e11*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(num0))
    # nuc0  = 1e15*(tdec/tsed)**(-(15.*g+24.)/(14.*g+7.)); print("{:.2e}".format(nuc0))
    # fnum0 = 94*(tdec/tsed)**(-(11.*g+12.)/(14.*g+7.)); print(fnum0, "\n")
    
    nua0 = 8.4e10
    num0 = 8.84e14
    nuc0 = 3.54e19
    fnum0 = 5.42e4
    
    #                     logFnumax,       lognumrs,     lognucutrs,       lognuars,   g, NOTE: logarithms are in base 10
    popt = np.array([np.log10(fnum0), np.log10(num0), np.log10(nuc0), np.log10(nua0),   g])
    
    plt.title(r"$\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, t_cross_small, *10**popt[:-1], 0, 0, k = k, p = pRS, g = popt[-1])
        nu_model, Fnu_model = modelFit.spectrum()
        #Fnu_model/np.sqrt(2) # TODO
        
        print("t   : {}".format(t))
        print("F   : {}".format(modelFit.Fnumaxrs()))
        print("num : {:.2e}".format(modelFit.numrs()))
        print("nua : {:.2e}".format(modelFit.nuars()))
        
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    #print("chi2:", "{:.2f}".format(chi2(fitFunc, tobs_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    #print("d.o.f.:", "{}".format(dof(fitFunc, tobs_grouped, Fnu_grouped)))
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/laskarFit", dpi = dpi)
    plt.close()
    
    plotBreakFreq("plots/fits/laskarFit", tobs_grouped, t_cross_small, *10**popt[:-1], 0, 0, p = pRS, k = k, g = popt[-1])
    
def plotZWZguess(tobs_grouped, nu_grouped, Fnu_grouped, err_FnuGrouped, colors_grouped, dpi = dpi):
    """"""
    tobs_mean = [np.mean(tobs_group) for tobs_group in tobs_grouped]
    
    def fitFunc(NU, logtcross, logFnumaxrs_tcross, lognumrs_tcross, lognucutrs_tcross, lognuars_tcross, pRS):
        """"""
        fsjettophat = [FSjetTopHat(nu_grouped[i], 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, tdays, 0.151, p = p, k = k) for i, tdays in enumerate(tobs_mean)]
        
        tcross = 10**logtcross
        Fnumaxrs_tcross = 10**logFnumaxrs_tcross
        numrs_tcross = 10**lognumrs_tcross
        nucutrs_tcross = 10**lognucutrs_tcross
        nuars_tcross = 10**lognuars_tcross
        
        FNU = []
        
        for i, tobs in enumerate(tobs_mean):
            model = RSjetStruct(tobs, NU[i], tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, 0.4, 2, k = 2, p = pRS, g = 1)
            nu_model, Fnu_model = model.spectrum()
            
            #FnuStack = np.hstack([FnuStack, Fnu_model])
            FNU.append(Fnu_model + fsjettophat[i].spectrum()[1])
            
        return FNU
    
    #              logtcross, logFnumax,  lognumrs, lognucutrs,  lognuars,  pRS. NOTE: logarithms are in base 10
    lowerBounds = [       -4,        -1,         5,         12,         5,    2]
    upperBounds = [        2,         7,        20,         20,        20,    3]
    p0 =          [       -3,         3,         7,         12,        10, 2.05]
    bounds =      [lowerBounds, upperBounds]
    popt, pcov = multiDimCurveFit(fitFunc, nu_grouped, Fnu_grouped, SIGMA = err_FnuGrouped, bounds = bounds, p0 = p0)
    print(*10**popt[:-1], popt[-1])
    
    plt.title(r"$\chi^2/d.o.f.$={:.2f}".format(chi2pdof(fitFunc, nu_grouped, Fnu_grouped, err_FnuGrouped, args = popt)))
    
    nu = np.geomspace(xmin, xmax, num = 100)
    
    for i, t in enumerate(LaskarTimes):
        modelFit = RSjetStruct(t, nu, *10**popt[:-1], 0.4, 2, k = k, p = popt[-1])
        nu_model, Fnu_model = modelFit.spectrum()
        plt.plot(nu_model, Fnu_model, linestyle = ":", color = getColor(tobs_grouped, 0, t = t))
        modelFS = FSjetTopHat(nu, 10**-0.616, 10**-0.221, 1, 10**-3.612, 10**54.04/1e52, t, 0.151, p = p, k = k)
        nu_FS, Fnu_FS = modelFS.spectrum()
        plt.plot(nu_FS, Fnu_FS, linestyle = "--", color = getColor(tobs_grouped, 0, t = t))
        plt.plot(nu_model, Fnu_model + Fnu_FS, color = getColor(tobs_grouped, 0, t = t), label = "{:.2f} d".format(LaskarTimes[i]))
    
    plt.legend(loc = "upper left")
    plt.savefig("plots/fits/ZWZ24guess", dpi = dpi)
    plt.close()
    
    plotBreakFreq("plots/fits/ZWZ24guess", tobs_grouped, *10**popt[:5], 0.4, 2, p = popt[5], k = k, g = 1)

def plotFanDiagriam(dpi = dpi):
    """Data taken from Laskar et al. 2023 table 7
    """
    t         = [3.46, 6.44, 12.44, 17.48, 28.33, 52.48, 76.42]
    Fpeak     = [8.6 , 4.9 ,  4.3 ,  2.9 ,  1.9 ,  1.2 ,  1.0 ]
    Fpeakerr  = [0.2 , 0.8 ,  0.8 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ]
    nupeak    = [1.67, 1.12,  1.19,  0.77,  0.56,  0.57,  0.45]
    nupeakerr = [0.02, 0.30,  0.25,  0.06,  0.03,  0.12,  0.07]
    
    Fpopt, Fpcov = sciopt.curve_fit(lcinterpolate.pl, t, Fpeak, sigma = Fpeakerr, p0 = [1, -1])
    _, dlnFpeakdlnt = Fpopt
    _, dlnFpeakdlnterr = np.sqrt(np.diag(Fpcov))

    nupopt, nupcov = sciopt.curve_fit(lcinterpolate.pl, t, nupeak, sigma = nupeakerr, p0 = [1, -1])
    _, dlnnudlnt = nupopt
    _, dlnnudlnterr = np.sqrt(np.diag(nupcov))
    
    tmin = 1 # 3e-1
    tmax = 1e2
    tannote1 = 10
    tannote2 = 0.9
    tannote3 = 1.45
    tannote4 = 7
    arrowprops = {"width"      : 0.05,
                  "headwidth"  : 3,
                  "headlength" : 5,
                  "shrink"     : 0.03,
                  "color"      : "k",
                  "alpha"      : 0.5}
    R = 0.55 # 1
    r = None
    tcont = np.geomspace(tmin, tmax, 1000)
    plt.title("Comparing Model Against Observed Features")
    plt.xscale("log")
    plt.xlabel("Time [days]")
    plt.yscale("log")
    #plt.ylabel(r"Peak Flux $F_{peak}$ [mJy]" + "\nor\n" + r"Peak Frequency $\nu_{peak}$ [GHz]")
    
    plotline, _, _ = plt.errorbar(t, Fpeak, yerr = Fpeakerr, fmt = "o", color = cbcolors[-1], label = r"obs $F_{peak}$")
    plotline.set_markerfacecolor("none")
    plt.plot(tcont, lcinterpolate.pl(tcont, *Fpopt), linestyle = ":", color = cbcolors[-1])
    
    xmax_Fobs, ymax_Fobs = plotlogfan(tmin, lcinterpolate.pl(tmin, *Fpopt), dlnFpeakdlnt, dlnFpeakdlnterr, R, color = cbcolors[-1], r = r, label = "observed", zorder = 1)
    xmax_Fstr, ymax_Fstr = plotlogfan(tmin, lcinterpolate.pl(tmin, *Fpopt), -0.75, 6.97e-2, R/1.35, color = cbcolors[-2], r = r, label = "structured", zorder = 2)
    xmax_Ftop, ymax_Ftop = plotlogfan(tmin, lcinterpolate.pl(tmin, *Fpopt), -1.02, 4.81e-2, R/2, color = cbcolors[-3], r = r, label = "top hat", zorder = 3)
    
    Fannote1 = lcinterpolate.pl(tannote1, *Fpopt)
    plt.annotate("Observed Peak\nFlux [mJy]", (tannote1, Fannote1), arrowprops = arrowprops, xytext = (2 * tannote1, 3 * Fannote1))
    #xytext_Fobs = (tannote4, 0.22 * ymax_Fobs); plt.annotate(        "Observed", (xmax_Fobs, ymax_Fobs), arrowprops = arrowprops, xytext = xytext_Fobs)
    xytext_Fstr = (tannote3, 0.3 * ymax_Fstr); plt.annotate("Structured\nModel", (xmax_Fstr, ymax_Fstr), arrowprops = arrowprops, xytext = xytext_Fstr)
    xytext_Ftop = (tannote2, 0.5 * ymax_Ftop); plt.annotate(   "Top Hat\nModel", (xmax_Ftop, ymax_Ftop), arrowprops = arrowprops, xytext = xytext_Ftop)
    
    plotline, _, _ = plt.errorbar(t, nupeak, yerr = nupeakerr, fmt = "d", color = cbcolors[-1], label = r"obs $\nu_{peak}$")
    plotline.set_markerfacecolor("none")
    plt.plot(tcont, lcinterpolate.pl(tcont, *nupopt), linestyle = ":", color = cbcolors[-1])
    
    xmax_nuobs, ymax_nuobs = plotlogfan(tmin, lcinterpolate.pl(tmin, *nupopt), dlnnudlnt, dlnnudlnterr, R, color = cbcolors[-1], r = r, zorder = 1)
    xmax_nustr, ymax_nustr = plotlogfan(tmin, lcinterpolate.pl(tmin, *nupopt), -1, 0, R/1.35, color = cbcolors[-2], r = r, zorder = 2)
    xmax_nutop, ymax_nutop = plotlogfan(tmin, lcinterpolate.pl(tmin, *nupopt), -1.66, 1.22e-1, R/2, color = cbcolors[-3], r = r, zorder = 3)
    
    nuannote1 = lcinterpolate.pl(tannote1, *nupopt)
    plt.annotate("Observed Peak\nFrequency [GHz]", (tannote1, nuannote1), arrowprops = arrowprops, xytext = (0.6 * tannote1, 0.3 * nuannote1))
    #plt.annotate(         "Observed", (xmax_nuobs, ymax_nuobs), arrowprops = arrowprops, xytext = xytext_Fobs)
    plt.annotate("Structured\nModel", (xmax_nustr, ymax_nustr), arrowprops = arrowprops, xytext = xytext_Fstr)
    plt.annotate(   "Top Hat\nModel", (xmax_nutop, ymax_nutop), arrowprops = arrowprops, xytext = xytext_Ftop)
    
    
    plt.gca().set_aspect("equal", "box")
    
    #handles, labels = plt.gca().get_legend_handles_labels()
    #order = [3, 4, 0, 1, 2]
    
    #plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    
    plt.savefig("plots/fits/FeatureComparison",  dpi = dpi)
    plt.savefig("plots/fits/FeatureComparison.svg")
    
#plotLCfits()
# plotErrorComparison("Peak Flux\nComparison", r"$d\ln F_{peak}/d\ln t$",\
#                     "plots/fits/peakFluxComparison",\
#                     [Fpeakobs, Fpeakobserr,   "observed", cbcolors[-1]],\
#                     [   -0.75,     6.97e-2, "structured", cbcolors[-2]],\
#                     [   -1.02,     4.81e-2,    "top hat", cbcolors[-3]],\
#                     bottom = -1.1)
# plotErrorComparison("Peak Frequency\nComparison", r"$d\ln \nu_{peak}/d\ln t$",\
#                     "plots/fits/peakFrequencyComparison",\
#                     [nupeakobs, nupeakobserr,   "observed", cbcolors[-1]],\
#                     [       -1,            0, "structured", cbcolors[-2]],\
#                     [    -1.66,      1.22e-1,    "top hat", cbcolors[-3]],\
#                     bottom = -1.93)
plotTopHatFitpRS(*plotData())
#plotFitpRS(*plotData())
plotFitpRSoneLessFree(*plotData())
#plotZWZguess(*plotData()) # TODO constrain to have crossing time before data
#plotFit(*plotData()) # TODO constrain to have crossing time before data
#plotLaskarFit(*plotData())
#plotFitpRSplusGG23(*plotData()) # TODO constrain to have crossing time before data
#plotFitpRSplusFreeFS(*plotData()) # TODO constrain to have crossing time before data
#plotFanDiagriam()
