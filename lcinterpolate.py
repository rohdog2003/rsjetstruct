# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:54:41 2025

@author: rohdo
"""

import numpy as np

from .gsspectshapes import Spectrum
from multidimcurvefit import chi2pdof, numParams, gof
import scipy.optimize as sciopt

def pl(x, F, a): # 2 params
     """"""
     return F * x**a
 
def pl_twoPoints(x, x12, y12):
    """"""
    x1 = x12[0]
    x2 = x12[1]
    y1 = y12[0]
    y2 = y12[1]
    
    m = (np.log(y2) - np.log(y1))/(np.log(x2) - np.log(x1))
    A = y1/x1**m
    
    return A * x**m

def bpl(x, xb, F, a1, a2, s): # 5 params
    sig = np.sign(a1-a2)
    s *= sig
    
    return Spectrum._Fnub(x, xb, F, a1, a2, s)

def dbpl(x, xb1, xb2, F, a1, a2, a3, s1, s2): # 8 params
    """"""
    sig1 = np.sign(a1-a2)
    sig2 = np.sign(a2-a3)
    
    s1 *= sig1
    s2 *= sig2
    
    return Spectrum._Fnub(x, xb1, F, a1, a2, s1) * Spectrum._tildeFnub(x, xb2, a2, a3, s2)

def tbpl(x, xb1, xb2, xb3, F, a1, a2, a3, a4, s1, s2, s3): # 11 params
    """"""
    sig1 = np.sign(a1-a2)
    sig2 = np.sign(a2-a3)
    sig3 = np.sign(a3-a4)
    
    s1 *= sig1
    s2 *= sig2
    s3 *= sig3
    
    return Spectrum._Fnub(x, xb1, F, a1, a2, s1) * Spectrum._tildeFnub(x, xb2, a2, a3, s2) * Spectrum._tildeFnub(x, xb3, a3, a4, s3)

def chi2pdofargmin(x): # TODO
    """gets the index of the earliest argument less than one or the index of 
    the smallest element if all are greater than one.
    """
    x = np.array(x)
    
    x[np.isnan(x)] = np.inf
    
    argm = 0
    m = x[0]
    
    for i, element in enumerate(x):
        if element < 1:
            return argm
        elif element < m:
            m = element
            argm = i
            
    return argm

def argclosest(x, ls):
    """"""
    ls = np.array(ls)
    
    return np.argmin(np.abs(x - ls))

def closest(x, ls):
    return ls[argclosest(x, ls)]

class LCinterpolate:
    
    def __init__(self, band, nu, t, F, err_F):
        """"""
        self._band = np.array(band)
        self._nu = np.array(nu)
        self._t = np.array(t)
        self._F = np.array(F)
        self._err_F = np.array(err_F)
    
    def xplfit(func, t, F, err_F):
        """"""
        N = numParams(func)
        M = {2 : 0, 5 : 1, 8 : 2, 11 : 3}[N] # number of breaks
        P = {2 : 1, 5 : 2, 8 : 3, 11 : 4}[N] # number of slopes
        
        breaks = np.geomspace(np.min(t), np.max(t), M)
        powers = np.flip(np.linspace(-2, 2, P))
        sharps = np.ones(N - M - P - 1)
        
        p0          = [                         *breaks, F[np.argmin(t)],                        *powers,                   *sharps]
        lowerBounds = [*np.full_like(breaks, np.min(t)),         -np.inf, *np.full_like(powers, -np.inf), *np.full_like(sharps,  0)]
        upperBounds = [*np.full_like(breaks, np.max(t)),          np.inf, *np.full_like(powers,  np.inf), *np.full_like(sharps, 10)]
        
        try:
            popt, pcov = sciopt.curve_fit(func, t, F, sigma = err_F, p0 = p0, bounds = [lowerBounds, upperBounds])
        except RuntimeError:
            popt = np.ones(N)
            pcov = np.zeros(N) 
    
        return popt, pcov
    
    def lcfit(self, x, bandToGet, return_gof = False):
        """"""
        _, t, F, err_F = self.getBandData(bandToGet)
        
        if len(t) == 1:
            lc = np.full_like(x, F[0])
            goodness = np.inf
        elif len(t) == 2:
            lc = pl_twoPoints(x, t, F)
            goodness = np.inf
        elif 3 <= len(t) <= 5:
            popt, _ = LCinterpolate.xplfit(pl, t, F, err_F)
            
            lc = pl(x, *popt)
            goodness = gof(pl, [t], [F], [err_F], args = popt)
        
        elif 6 <= len(t) <= 8:
            popt1, _ = LCinterpolate.xplfit(pl, t, F, err_F)
            popt2, _ = LCinterpolate.xplfit(bpl, t, F, err_F)
            
            chi2pdof1 = chi2pdof(pl, [t], [F], [err_F], args = popt1)#; print(chi2pdof1) # TODO
            chi2pdof2 = chi2pdof(bpl, [t], [F], [err_F], args = popt2)#; print(chi2pdof2) # TODO
            
            minum = chi2pdofargmin([chi2pdof1, chi2pdof2]) + 1#; print(minum, "\n") # TODO   
            
            if minum == 1:
                lc = pl(x, *popt1)
                goodness = gof(pl, [t], [F], [err_F], args = popt1)
            elif minum == 2:
                lc = bpl(x, *popt2)
                goodness = gof(bpl, [t], [F], [err_F], args = popt2)
            
        elif 9 <= len(t) <= 11:
            popt1, _ = LCinterpolate.xplfit(pl, t, F, err_F)
            popt2, _ = LCinterpolate.xplfit(bpl, t, F, err_F)
            popt3, _ = LCinterpolate.xplfit(dbpl, t, F, err_F)
            
            chi2pdof1 = chi2pdof(pl, [t], [F], [err_F], args = popt1)#; print(chi2pdof1) # TODO
            chi2pdof2 = chi2pdof(bpl, [t], [F], [err_F], args = popt2)#; print(chi2pdof2) # TODO
            chi2pdof3 = chi2pdof(dbpl, [t], [F], [err_F], args = popt3)#; print(chi2pdof3, "\n") # TODO
            
            minum = chi2pdofargmin([chi2pdof1, chi2pdof2, chi2pdof3]) + 1
            
            if minum == 1:
                lc = pl(x, *popt1)
                goodness = gof(pl, [t], [F], [err_F], args = popt1)
            elif minum == 2:
                lc = bpl(x, *popt2)
                goodness = gof(bpl, [t], [F], [err_F], args = popt2)
            elif minum == 3:
                lc = dbpl(x, *popt3)
                goodness = gof(dbpl, [t], [F], [err_F], args = popt3)
            
        elif 12 <= len(t):
            popt1, _ = LCinterpolate.xplfit(pl, t, F, err_F)
            popt2, _ = LCinterpolate.xplfit(bpl, t, F, err_F)
            popt3, _ = LCinterpolate.xplfit(dbpl, t, F, err_F)
            popt4, _ = LCinterpolate.xplfit(tbpl, t, F, err_F)
            
            chi2pdof1 = chi2pdof(pl, [t], [F], [err_F], args = popt1)
            chi2pdof2 = chi2pdof(bpl, [t], [F], [err_F], args = popt2)
            chi2pdof3 = chi2pdof(dbpl, [t], [F], [err_F], args = popt3)
            chi2pdof4 = chi2pdof(tbpl, [t], [F], [err_F], args = popt4)
            
            minum = chi2pdofargmin([chi2pdof1, chi2pdof2, chi2pdof3, chi2pdof4]) + 1
            
            if minum == 1:
                lc = pl(x, *popt1)
                goodness = gof(pl, [t], [F], [err_F], args = popt1)
            elif minum == 2:
                lc = bpl(x, *popt2)
                goodness = gof(bpl, [t], [F], [err_F], args = popt2)
            elif minum == 3:    
                lc = dbpl(x, *popt3)
                goodness = gof(dbpl, [t], [F], [err_F], args = popt3)
            elif minum == 4:
                lc = tbpl(x, *popt4)
                goodness = gof(tbpl, [t], [F], [err_F], args = popt4)
            
        if not(return_gof):
            return lc
        else:
            return lc, goodness
    
    def getBandData(self, bandToGet):
        """"""
        nu = self._nu[self._band == bandToGet]
        t = self._t[self._band == bandToGet]
        F = self._F[self._band == bandToGet]
        err_F = self._err_F[self._band == bandToGet]
    
        return nu, t, F, err_F
    
    def lcinterpolate(self, t_interp):
        """"""
        bands_unique = np.unique(self._band)
        
        tobs_interp = np.zeros_like(self._t)
        F_interp = np.zeros_like(self._F)
        err_F_interp = np.zeros_like(self._err_F)
        
        for b in bands_unique:
            argband = np.array([i for i in range(len(self._band))])[self._band == b]
            
            t_band = self._t[self._band == b]
            t_closest = [closest(t, t_interp) for t in [self._t[i] for i in argband]]
            
            F_band, F_closest = self.lcfit(np.array([t_band, t_closest]), b)
            
            factor = F_closest/F_band
            
            for j, i in enumerate(argband):
                tobs_interp[i] = t_closest[j]
                F_interp[i] = self._F[i] * factor[j]
                err_F_interp[i] = self._err_F[i] * factor[j]
                
        return self._nu, tobs_interp, F_interp, err_F_interp