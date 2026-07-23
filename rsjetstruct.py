# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:19:34 2025

@author: rohdo
"""
import numpy as np
from .gsspectshapes import Spectrum
#from obsfluxmax import *
import warnings
from .utilities import where # TODO implement this as needed

smallNum = 1e-50 # a very small number close to zero
largeTime = 1e7 # 1000000


# ---- Vectorised GS02 spectrum helpers ------------------------------------
# Module-level array-aware ports of the formulas in rsjetstruct.gsspectshapes.
# The Spectrum class assumed scalar break frequencies and was called once
# per (t, nu) point via @np.vectorize, which dominated profiling. These
# helpers operate on full-length arrays so the whole RS spectrum is
# computed in a handful of vectorised numpy calls.

def _slope_v(b, p, k):
    """Mirrors Spectrum._getSlope. Called at most O(1) times per RS spectrum."""
    pick = (lambda a, c: a if k == 0 else c)
    if b == 1:  return 2.0,         1.0/3.0,        pick(1.64,            1.06)
    if b == 2:  return 1.0/3.0,     (1.0 - p)/2.0,  pick(1.84 - 0.40*p,   1.76 - 0.38*p)
    if b == 3:  return (1.0 - p)/2.0, -p/2.0,       pick(1.15 - 0.06*p,   0.80 - 0.03*p)
    if b == 4:  return 2.0,         5.0/2.0,        pick(3.44*p - 1.41,   3.63*p - 1.60)
    if b == 5:  return 5.0/2.0,     (1.0 - p)/2.0,  pick(1.47 - 0.21*p,   1.25 - 0.18*p)
    if b == 6:  return 5.0/2.0,     -p/2.0,         pick(0.94 - 0.14*p,   1.04 - 0.16*p)
    if b == 7:  return 2.0,         11.0/8.0,       pick(1.99 - 0.04*p,   1.97 - 0.04*p)
    if b == 8:  return 11.0/8.0,    -1.0/2.0,       pick(0.907,           0.893)
    if b == 9:  return -1.0/2.0,    -p/2.0,         pick(3.34 - 0.82*p,   3.68 - 0.89*p)
    if b == 10: return 11.0/8.0,    1.0/3.0,        pick(1.213,           1.213)
    if b == 11: return 1.0/3.0,     -1.0/2.0,       pick(0.597,           0.597)
    if b in (12, 13, 14, 15):
        return float("nan"), float("nan"), pick(2.0, 2.0)
    raise ValueError("slope index out of range: %r" % (b,))


def _Fnub_v(nu, nub, Fnub, beta1, beta2, s):
    """GS02 (1)."""
    r = nu / nub
    return Fnub * (r**(-s * beta1) + r**(-s * beta2))**(-1.0 / s)


def _Fnu4_v(nu, nu4, Fnu4, beta1, beta2, s):
    """GS02 (3). beta1/beta2 unused; matches Spectrum._Fnu4 signature."""
    phi4 = nu / nu4
    return Fnu4 * (phi4**2 * np.exp(-s * phi4**(2.0/3.0)) + phi4**(5.0/2.0))


def _tildeFnub_v(nu, nub, beta1, beta2, s):
    """GS02 (4)."""
    return (1.0 + (nu/nub)**(s * (beta1 - beta2)))**(-1.0 / s)


def _tildeFnuCUT12_v(nu, nuc, p, k):
    """Array-aware port of Spectrum._tildeFnuCUT12 (cutoff for spectra 1,2)."""
    s12 = _slope_v(12, p, k)[2]
    slope3 = _slope_v(3, p, k)
    mask = nu / nuc < 7e2
    safe = np.where(mask, nu / nuc, 0.0)
    tildeFnu3 = _tildeFnub_v(nu, nuc, *slope3)
    tildeFnu3atnu3 = _tildeFnub_v(nuc, nuc, *slope3)
    inner = (tildeFnu3**(-s12)
             + tildeFnu3atnu3**(-s12) * np.exp(-s12) * (np.exp(s12 * safe) - 1.0))**(-1.0 / s12)
    return np.where(mask, inner, 0.0)


def _FnuCUT3_v(nu, Fnu4, nusa, num, nuc, p, k):
    """Array-aware port of Spectrum._FnuCUT3 (cutoff for spectrum 3)."""
    s13 = _slope_v(13, p, k)[2]
    slope4 = _slope_v(4, p, k)
    slope6 = _slope_v(6, p, k)
    mask = nu / nuc < 7e2
    safe = np.where(mask, nu / nuc, 0.0)
    precut = _Fnu4_v(nu, num, Fnu4, *slope4) * _tildeFnub_v(nu, nusa, *slope6)
    # Mirrors the upstream literal: _Fnu4 evaluated at nuc, _tildeFnub at nu.
    precutatcut = _Fnu4_v(nuc, num, Fnu4, *slope4) * _tildeFnub_v(nu, nusa, *slope6)
    inner = (precut**(-s13)
             + precutatcut**(-s13) * np.exp(-s13) * (np.exp(s13 * safe) - 1.0))**(-1.0 / s13)
    return np.where(mask, inner, 0.0)


def _FnuCUT4_v(nu, Fnu7, nuac, nusa, num, nuc, p, k):
    """Array-aware port of Spectrum._FnuCUT4 (cutoff for spectrum 4)."""
    s14 = _slope_v(14, p, k)[2]
    slope7 = _slope_v(7, p, k)
    slope8 = _slope_v(8, p, k)
    slope9 = _slope_v(9, p, k)
    mask = nu / nuc < 7e2
    safe = np.where(mask, nu / nuc, 0.0)
    precut = (_Fnub_v(nu, nuac, Fnu7, *slope7)
              * _tildeFnub_v(nu, nusa, *slope8)
              * _tildeFnub_v(nu, num, *slope9))
    precutatnu11 = (_Fnub_v(nuc, nuac, Fnu7, *slope7)
                    * _tildeFnub_v(nuc, nusa, *slope8)
                    * _tildeFnub_v(nuc, num, *slope9))
    inner = (precut**(-s14)
             + precutatnu11**(-s14) * np.exp(-s14) * (np.exp(s14 * safe) - 1.0))**(-1.0 / s14)
    return np.where(mask, inner, 0.0)


def _tildeFnuCUT5_v(nu, nuc, p, k):
    """Array-aware port of Spectrum._tildeFnuCUT5 (cutoff for spectrum 5)."""
    s15 = _slope_v(15, p, k)[2]
    slope11 = _slope_v(11, p, k)
    mask = nu / nuc < 7e2
    safe = np.where(mask, nu / nuc, 0.0)
    tildeFnu11 = _tildeFnub_v(nu, nuc, *slope11)
    tildeFnu11atnu11 = _tildeFnub_v(nuc, nuc, *slope11)
    inner = (tildeFnu11**(-s15)
             + tildeFnu11atnu11**(-s15) * np.exp(-s15) * (np.exp(s15 * safe) - 1.0))**(-1.0 / s15)
    return np.where(mask, inner, 0.0)


def _classify_branch_v(nuac, nusa, num, nuc):
    """Vectorised port of the Spectrum.spectrum() if/elif chain.

    nuac is the absorption-coefficient floor (scalar smallNum in the RS path).
    Assigns in reverse so earlier branches (matching the original if/elif
    short-circuit order) win when multiple conditions hold.
    """
    n = nusa.shape[0]
    branch = np.zeros(n, dtype=np.int8)
    # spectrum 5: nuac <= nusa <= nuc <= num
    branch[(nuac <= nusa) & (nusa <= nuc) & (nuc <= num)] = 5
    # spectrum 4: nuac <= nusa AND nuc <= nusa AND nusa <= num
    branch[(nuac <= nusa) & (nuc <= nusa) & (nusa <= num)] = 4
    # spectrum 3: nuac <= nusa AND num <= nusa AND nuc <= nusa
    branch[(nuac <= nusa) & (num <= nusa) & (nuc <= nusa)] = 3
    # spectrum 2: nuac <= num <= nusa <= nuc
    branch[(nuac <= num) & (num <= nusa) & (nusa <= nuc)] = 2
    # spectrum 1: nuac <= nusa <= num <= nuc  (highest priority -> assigned last)
    branch[(nuac <= nusa) & (nusa <= num) & (num <= nuc)] = 1
    return branch


def _obs_flux_max_v(Fnumax, nusa, num, nuc, p, branch, specnum_forced=None):
    """Vectorised port of obsFluxMax. `branch` already encodes specnum per point.

    The upstream obsFluxMax uses an activation gate
        `cond_N * (specnum is None) + (specnum == N)`
    on each branch's formula. In auto (specnum=None) mode the gate selects
    the matching branch; under a forced specnum the gate forces that branch's
    formulas to fire regardless of break ordering. Spectrum 3 is special: it
    has two sub-formulas (3a: num<=nuc, 3b: num>nuc) each gated independently.
    Under forced specnum=3 both sub-formulas activate and the result is their
    sum; under auto mode the two are mutually exclusive on num vs nuc.
    """
    F = np.zeros_like(Fnumax)
    is1 = branch == 1
    is2 = branch == 2
    is3 = branch == 3
    is4 = branch == 4
    is5 = branch == 5
    F[is1] = Fnumax[is1]
    F[is2] = Fnumax[is2] * (nusa[is2] / num[is2])**(-(p - 1.0)/2.0)
    if specnum_forced == 3:
        # Forced spec 3: the upstream gate `cond * 0 + 1` makes both sub-cases
        # fire unconditionally, so the result is their sum.
        F[is3] = (Fnumax[is3]
                  * (nuc[is3] / num[is3])**(-(p - 1.0)/2.0)
                  * (nusa[is3] / nuc[is3])**(-p/2.0)
                + Fnumax[is3]
                  * (num[is3] / nuc[is3])**(-0.5)
                  * (nusa[is3] / num[is3])**(-p/2.0))
    else:
        is3a = is3 & (num <= nuc)
        is3b = is3 & (num >  nuc)
        F[is3a] = (Fnumax[is3a]
                   * (nuc[is3a] / num[is3a])**(-(p - 1.0)/2.0)
                   * (nusa[is3a] / nuc[is3a])**(-p/2.0))
        F[is3b] = (Fnumax[is3b]
                   * (num[is3b] / nuc[is3b])**(-0.5)
                   * (nusa[is3b] / num[is3b])**(-p/2.0))
    F[is4] = Fnumax[is4] * (nusa[is4] / nuc[is4])**(-0.5)
    F[is5] = Fnumax[is5]
    return F


def _spectrum_branch_v(sn, nu, Fnumax, nuac, nusa, num, nuc, p, k, cut):
    """Vectorised computation of GS02 spectrum branch `sn` for arrays.

    `Fnumax` is the obs_flux_max'd peak flux (the same `_Fnutruemaxrs` the
    scalar Spectrum class consumed). `cut` is the boolean array selecting
    post-tcross points (or the override).
    """
    not_cut = ~cut
    if sn == 1:
        beta1_2 = _slope_v(2, p, k)[0]
        Fnu1 = Fnumax * (nusa / num)**beta1_2
        return (_Fnub_v(nu, nusa, Fnu1, *_slope_v(1, p, k))
                * _tildeFnub_v(nu, num, *_slope_v(2, p, k))
                * (not_cut * _tildeFnub_v(nu, nuc, *_slope_v(3, p, k))
                   + cut * _tildeFnuCUT12_v(nu, nuc, p, k)))
    if sn == 2:
        beta1_5 = _slope_v(5, p, k)[0]
        Fnu4 = Fnumax * (num / nusa)**beta1_5
        return (_Fnu4_v(nu, num, Fnu4, *_slope_v(4, p, k))
                * _tildeFnub_v(nu, nusa, *_slope_v(5, p, k))
                * (not_cut * _tildeFnub_v(nu, nuc, *_slope_v(3, p, k))
                   + cut * _tildeFnuCUT12_v(nu, nuc, p, k)))
    if sn == 3:
        beta1_5 = _slope_v(5, p, k)[0]
        Fnu4 = Fnumax * (num / nusa)**beta1_5
        return (not_cut * _Fnu4_v(nu, num, Fnu4, *_slope_v(4, p, k))
                        * _tildeFnub_v(nu, nusa, *_slope_v(6, p, k))
                + cut * _FnuCUT3_v(nu, Fnu4, nusa, num, nuc, p, k))
    if sn == 4:
        beta1_8 = _slope_v(8, p, k)[0]
        # nuac is scalar smallNum in the RS path; broadcasts elementwise.
        Fnu7 = Fnumax * (nuac / nusa)**beta1_8
        return (not_cut * _Fnub_v(nu, nuac, Fnu7, *_slope_v(7, p, k))
                        * _tildeFnub_v(nu, nusa, *_slope_v(8, p, k))
                        * _tildeFnub_v(nu, num, *_slope_v(9, p, k))
                + cut * _FnuCUT4_v(nu, Fnu7, nuac, nusa, num, nuc, p, k))
    if sn == 5:
        beta1_11 = _slope_v(11, p, k)[0]
        beta1_10 = _slope_v(10, p, k)[0]
        Fnu7 = Fnumax * (nusa / nuc)**beta1_11 * (nuac / nusa)**beta1_10
        return (_Fnub_v(nu, nuac, Fnu7, *_slope_v(7, p, k))
                * _tildeFnub_v(nu, nusa, *_slope_v(10, p, k))
                * (not_cut * _tildeFnub_v(nu, nuc, *_slope_v(11, p, k))
                            * _tildeFnub_v(nu, num, *_slope_v(9, p, k))
                   + cut * _tildeFnuCUT5_v(nu, nuc, p, k)))
    raise ValueError("unknown spectrum branch: %r" % (sn,))

def obsFluxMax(Fnumax_nossa, nuac, nusa, num, nuc, p, specnum = None): # TODO add by specnum?
    """computes the observed maximum flux from the theoretical maximum if no
    synchrotron self absorption were to occur.
    """
    F1 = Fnumax_nossa * ((nuac <= nusa <= num <= nuc) * (specnum is None) + (specnum == 1)) # spectrum 1
    F2 = Fnumax_nossa * (nusa/num)**(-(p - 1)/2) * ((nuac <= num <= nusa <= nuc) * (specnum is None) + (specnum == 2))  # spectrum 2
    F3 = Fnumax_nossa * (nuc/num)**(-(p - 1)/2) * (nusa/nuc)**(-p/2) * ((nuac <= nusa and num <= nusa and nuc <= nusa and num <= nuc) * (specnum is None) + (specnum == 3)) + \
         Fnumax_nossa * (num/nuc)**(-1/2) * (nusa/num)**(-p/2) * ((nuac <= nusa and num <= nusa and nuc <= nusa and num > nuc) * (specnum is None) + (specnum == 3))  # spectrum 3
    F4 = Fnumax_nossa * (nusa/nuc)**(-1/2) * ((nuac <= nusa and nuc <= nusa and nusa <= num) * (specnum is None) + (specnum == 4)) # spectrum 4
    F5 = Fnumax_nossa * ((nuac <= nusa <= nuc <= num) * (specnum is None) + (specnum == 5)) # spectrum 5
    if (sum(np.array([nuac > nusa])) > 0):
        warnings.warn("nuac must be smaller than nusa", RuntimeWarning)
    if (sum(np.array([F1, F2, F3, F4, F5])) == 0):
        warnings.warn("No cases satisfied in obsFluxMax, flux returned is zero! Check input parameters.", RuntimeWarning)
    return F1 + F2 + F3 + F4 + F5

    # if nuac <= nusa <= num <= nuc: # spectrum 1
    #     return Fnumax_nossa
    # elif nuac <= num <= nusa <= nuc: # spectrum 2
    #     return Fnumax_nossa * (nusa/num)**(-(p - 1)/2)
    # elif nuac <= nusa and num <= nusa and nuc <= nusa: # spectrum 3
    #     if num <= nuc:
    #         return Fnumax_nossa * (nuc/num)**(-(p - 1)/2) * (nusa/nuc)**(-p/2)
    #     else:
    #         return Fnumax_nossa * (num/nuc)**(-1/2) * (nusa/num)**(-p/2)
    # elif nuac <= nusa and nuc <= nusa and nusa <= num: # spectrum 4
    #     return Fnumax_nossa * (nusa/nuc)**(-1/2)
    # elif nuac <= nusa <= nuc <= num: # spectrum 5 first case
    #     return Fnumax_nossa
    # else:
    #     raise Exception("nuac must be smaller than nusa")

class RSjetStruct:
    """Zhang, Weng, and Zheng 2024 (ZWZ24) which assumes slow cooling in the 
    thin shell case. An important observation if there is slow cooling at 
    crossing time then there is slow cooling pre crossing time.
    
    Note : break frequency parameters should not be equal
    """
    
    def __init__(self, tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps = 0, kGamma = 0,\
                 k = 0, p = 2.5, g = None, tjet = np.inf, weighted = True, tNRFS = np.inf):
        """Constructor
        
        Parameters
        ----------
        tobs : ndarray
            The observer times for which to calculate values.
        nu : ndarray
            The frequencies for which to calculate values.
        tcross : ndarray
            The crossing time.
        Fnumaxrs_tcross : ndarray
            The maximum flux before SSA corrections at crossing time. 
        numrs_tcross : ndarray
            The minimum break frequency at crossing time.
        nucutrs_tcross : ndarray
            The cooling/cut break frequency at crossing time.
        nuars_tcross : ndarray
            The absorption break frequency at crossing time.
        keps : float
            The power law of the energy per solid angle wing as a function of 
            observer angle. 
            TODO is it observer angle?
        kGamma : float
            The power law of the initial Lorentz factor wing as a 
            function of observer angle.
            TODO is it observer angle?
        p : float
            The power law of the injection distribution as a function of 
            electron Lorentz factor. Between 2 (inclusive) and 3 (inclusive).
        k : int
            Is 0 for ISM and 2 for wind
        g : float, default = None
            The power law of the initial Lorentz factor as a function of radius/(deceleration radius). 
        tjet : float, default = np.inf
            The break time of the jet.
        weighted : bool, default = True
            If True then weight the spectra depending on the crossing times.
        
        NOTE: due to scaling output will have the same units as the given input
        """ 
        self._tobs = tobs
        self._nu = nu
        self._tcross = tcross
        self._Fnumaxrs_tcross = Fnumaxrs_tcross
        self._numrs_tcross = numrs_tcross
        self._nucutrs_tcross = nucutrs_tcross
        self._nuars_tcross = nuars_tcross
        self._keps = keps
        self._kGamma = kGamma
        self._p = p
        self._k = k
        self._ISM = RSjetStruct._ISM(k)
        self._g = RSjetStruct._g(self._ISM, g)
        self._tjet = tjet
        self._weighted = weighted
        self._tNRFS = tNRFS
        
        self._tfrac = self._tobs/self._tcross
        self._a = self._compute_a()
        self._A = self._compute_A()
        
        self._Gamma3alphaDict = self._buildGamma3alphaDict()
        self._alphaDict = self._buildAlphaDict()
        # post crossing time equalities for nuars and numrs slow cooling 
        # TODO add case c
        if self._ISM:
            self._tnuarseqnumrsPostCrossISMcaseIa     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIa"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
            self._tnuarseqnumrsPostCrossISMcaseIb     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
            self._tnuarseqnumrsPostCrossISMcaseIIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIIa"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
            self._tnuarseqnumrsPostCrossISMcaseIIb    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
        else:
            self._tnuarseqnumrsPostCrossISMcaseIa     =       np.nan
            self._tnuarseqnumrsPostCrossISMcaseIb     =       np.nan
            self._tnuarseqnumrsPostCrossISMcaseIIa    =       np.nan
            self._tnuarseqnumrsPostCrossISMcaseIIb    =       np.nan
            
        if not(self._ISM):
            self._tnuarseqnumrsPostCrossWindCaseIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIa"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
            self._tnuarseqnumrsPostCrossWindCaseIb    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
            self._tnuarseqnumrsPostCrossWindCaseIIa   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIIa"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
            self._tnuarseqnumrsPostCrossWindCaseIIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
        else:
            self._tnuarseqnumrsPostCrossWindCaseIa    =       np.nan
            self._tnuarseqnumrsPostCrossWindCaseIb    =       np.nan
            self._tnuarseqnumrsPostCrossWindCaseIIa   =       np.nan
            self._tnuarseqnumrsPostCrossWindCaseIIb   =       np.nan
            
            
        # post crossing time equalities for nuars and nucutrs slow cooling
        if self._ISM:
            self._tnuarseqnucutrsPostCrossISMcaseIa   = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIa"],\
                                                                                                     self._tnuarseqnumrsPostCrossISMcaseIa,      self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
            self._tnuarseqnucutrsPostCrossISMcaseIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
            self._tnuarseqnucutrsPostCrossISMcaseIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIc"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
            self._tnuarseqnucutrsPostCrossISMcaseIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIa"],\
                                                                                                     self._tnuarseqnumrsPostCrossISMcaseIIa,     self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
            self._tnuarseqnucutrsPostCrossISMcaseIIb  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
            self._tnuarseqnucutrsPostCrossISMcaseIIc  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIc"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
        else:
            self._tnuarseqnucutrsPostCrossISMcaseIa   =       np.nan
            self._tnuarseqnucutrsPostCrossISMcaseIb   =       np.nan
            self._tnuarseqnucutrsPostCrossISMcaseIc   =       np.nan
            self._tnuarseqnucutrsPostCrossISMcaseIIa  =       np.nan
            self._tnuarseqnucutrsPostCrossISMcaseIIb  =       np.nan
            self._tnuarseqnucutrsPostCrossISMcaseIIc  =       np.nan
        
        if not(self._ISM):
            self._tnuarseqnucutrsPostCrossWindCaseIa   = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIa"],\
                                                                                                      self._tnuarseqnumrsPostCrossWindCaseIa,    self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
            self._tnuarseqnucutrsPostCrossWindCaseIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
            self._tnuarseqnucutrsPostCrossWindCaseIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIc"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
            self._tnuarseqnucutrsPostCrossWindCaseIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIa"],\
                                                                                                      self._tnuarseqnumrsPostCrossWindCaseIIa,   self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
            self._tnuarseqnucutrsPostCrossWindCaseIIb  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
            self._tnuarseqnucutrsPostCrossWindCaseIIc  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIc"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
        else:
            self._tnuarseqnucutrsPostCrossWindCaseIa   =       np.nan
            self._tnuarseqnucutrsPostCrossWindCaseIb   =       np.nan
            self._tnuarseqnucutrsPostCrossWindCaseIc   =       np.nan
            self._tnuarseqnucutrsPostCrossWindCaseIIa  =       np.nan
            self._tnuarseqnucutrsPostCrossWindCaseIIb  =       np.nan
            self._tnuarseqnucutrsPostCrossWindCaseIIc  =       np.nan
            
        # post crossing time equalities for nuars and numrs slow cooling (double crossing)
        if self._ISM:
            self._tnuarseqnumrsPostCrossISMcaseIc      = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIc"],\
                                                                                                      self._tnuarseqnucutrsPostCrossISMcaseIc,   self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
            self._tnuarseqnumrsPostCrossISMcaseIIc     = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIIc"],\
                                                                                                      self._tnuarseqnucutrsPostCrossISMcaseIIc,  self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
        else:
            self._tnuarseqnumrsPostCrossISMcaseIc      = np.nan
            self._tnuarseqnumrsPostCrossISMcaseIIc     = np.nan
        
        if not(self._ISM):
            self._tnuarseqnumrsPostCrossWindCaseIc     = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIc"],\
                                                                                                      self._tnuarseqnucutrsPostCrossWindCaseIc,  self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
            self._tnuarseqnumrsPostCrossWindCaseIIc    = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIIc"],\
                                                                                                      self._tnuarseqnucutrsPostCrossWindCaseIIc, self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
        else:
            self._tnuarseqnumrsPostCrossWindCaseIc     = np.nan
            self._tnuarseqnumrsPostCrossWindCaseIIc    = np.nan
            
        # pre crossing time equalities for nuars and numrs slow cooling
        if self._ISM:
            self._tnuarseqnumrsPreCrossISMcaseIIIa     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIIIa"],  self._alphaDict["numrs"]["ISMcaseIII"],    postcross = False)
            self._tnuarseqnumrsPreCrossISMcaseIIIb     =       np.nan # always np.nan
            self._tnuarseqnumrsPreCrossISMcaseIIIc     =       np.nan # always np.nan
        else:
            self._tnuarseqnumrsPreCrossISMcaseIIIa     =       np.nan
            self._tnuarseqnumrsPreCrossISMcaseIIIb     =       np.nan
            self._tnuarseqnumrsPreCrossISMcaseIIIc     =       np.nan
        
        if not(self._ISM):
            self._tnuarseqnumrsPreCrossWindCaseIIIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIIIa"], self._alphaDict["numrs"]["windCaseIII"],   postcross = False)
            self._tnuarseqnumrsPreCrossWindCaseIIIb    =       np.nan # always np.nan
            self._tnuarseqnumrsPreCrossWindCaseIIIc    =       np.nan # always np.nan
        else:
            self._tnuarseqnumrsPreCrossWindCaseIIIa    =       np.nan
            self._tnuarseqnumrsPreCrossWindCaseIIIb    =       np.nan
            self._tnuarseqnumrsPreCrossWindCaseIIIc    =       np.nan
            
        # pre crossing time equalities for nuars and nucutrs slow cooling
        if self._ISM:
            self._tnuarseqnucutrsPreCrossISMcaseIIIa   =       np.nan # always np.nan
            self._tnuarseqnucutrsPreCrossISMcaseIIIb   =       np.nan # always np.nan
            self._tnuarseqnucutrsPreCrossISMcaseIIIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["ISMcaseIIIc"],  self._alphaDict["nucutrs"]["ISMcaseIII"],  postcross = False)
        else:
            self._tnuarseqnucutrsPreCrossISMcaseIIIa   =       np.nan 
            self._tnuarseqnucutrsPreCrossISMcaseIIIb   =       np.nan
            self._tnuarseqnucutrsPreCrossISMcaseIIIc   =       np.nan
        
        if not(self._ISM):
            self._tnuarseqnucutrsPreCrossWindCaseIIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIIa"],\
                                                                                                      self._tnuarseqnumrsPreCrossWindCaseIIIa,   self._alphaDict["nuars"]["windCaseIIIb"], self._alphaDict["nucutrs"]["windCaseIII"], postcross = False) # FIXME sometimes np.nan when it shouldn't be
            self._tnuarseqnucutrsPreCrossWindCaseIIIb =        RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIIb"], self._alphaDict["nucutrs"]["windCaseIII"], postcross = False)
            self._tnuarseqnucutrsPreCrossWindCaseIIIc =        np.nan # always np.nan
        else:
            self._tnuarseqnucutrsPreCrossWindCaseIIIa =        np.nan
            self._tnuarseqnucutrsPreCrossWindCaseIIIb =        np.nan
            self._tnuarseqnucutrsPreCrossWindCaseIIIc =        np.nan
        
        self._Fnumaxrs = self.Fnumaxrs()
        self._numrs = self.numrs()
        self._nucutrs = self.nucutrs()
        self._nuars = self.nuars()
    
    def spectrum(self, diagnostic=False):
        """Returns the frequncy and fluxes at those frequencies for a Granot 
        and Sari 2002 spectra.
        
        Parameters
        ----------
        diagnostic: bool, default = False
            If True then return break frequencies and peak flux instead.
        
        Returns
        -------
        Fnu : float
            Fluxes
        """
        if diagnostic:
            return RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, diagnostic=True, specnum = None)
        if not(self._weighted):
            return RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = None)
        else:
            return self._spectrumweighted()
            
    def _spectrumweighted(self):
        """Spectrum weighting for slow cooling"""
        pl = 2
        firstweight = lambda x, x0, power : 1. / (1. + np.clip(x/x0, a_min=None, a_max=1e10)**power)
        
        if self._ISM: # ISM
            if (self._nuars_tcross <= self._numrs_tcross <= self._nucutrs_tcross): # spectrum 1              
                # crossing times                    
                tx1 = self._tnuarseqnumrsPreCrossISMcaseIIIa
                
                if self._kGamma <= 1:
                    tx2 = self._tnuarseqnumrsPostCrossISMcaseIa
                    tx3 = self._tnuarseqnucutrsPostCrossISMcaseIa
                else:
                    tx2 = self._tnuarseqnumrsPostCrossISMcaseIIa
                    tx3 = self._tnuarseqnucutrsPostCrossISMcaseIIa
                
                # weights 
                w21 = firstweight(self._tobs,tx1,15) # first indice spectrum number, second indice pre (1) or post (2) deceleration             
                w11 = 1./(1.+(self._tobs/tx1)**(-1.*15))
                w12a = 1./(1.+(self._tobs/tx2)**pl)
                w22a = 1./(1.+(self._tobs/tx2)**(-1.*pl))
                w22b = 1/(1+(self._tobs/tx3)**pl)
                w32b = 1/(1+(self._tobs/tx3)**(-pl))
                
                # spectra
                y21 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = False) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y11 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = False) # calc_spect(1, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y12 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = True) # calc_spect(1, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True) 
                
                if tx2 is np.nan or tx2 >= largeTime: # if no second crossing then always in spectrum 1 post deceleration
                    spect = (w21*y21 + w11*y11)/(w21+w11)*(self._tobs < self._tcross) + y12*(self._tobs >= self._tcross)
                elif tx3 is np.nan or tx3 >= largeTime: # if nuars goes above num but not above nucut post deceleration
                    spect = (w21*y21 + w11*y11)/(w21+w11)*(self._tobs < self._tcross) + (w12a*y12+w22a*y22)/(w12a+w22a)*(self._tobs >= self._tcross)   
                else: # if nuars goes above both num and nucut post deceleration
                    # weighted average
                    spect =  (w21*y21 + w11*y11)/(w21+w11)*(self._tobs < self._tcross) + (w12a*y12+(w22a*w22b)*y22+w32b*y32)/(w12a+w22a*w22b+w32b)*(self._tobs >= self._tcross)   
                
            elif (self._numrs_tcross <= self._nuars_tcross <= self._nucutrs_tcross):  # spectrum 2
                # Calculate tx1
                if self._kGamma <= 1:
                    tx1up = self._tnuarseqnucutrsPostCrossISMcaseIb # (self._nucutrs_tcross/self._nuars_tcross)**(1/(laE2-lc2))*self._tcross                    
                    tx1down = self._tnuarseqnumrsPostCrossISMcaseIb
                else:
                    tx1up = self._tnuarseqnucutrsPostCrossISMcaseIIb
                    tx1down = self._tnuarseqnumrsPostCrossISMcaseIIb
                
                w22a = 1/(1+(self._tobs/tx1down)**pl) 
                w12a = 1/(1+(self._tobs/tx1down)**(-pl))
                w22b = 1/(1+(self._tobs/tx1up)**pl) 
                w32b = 1/(1+(self._tobs/tx1up)**(-pl))
                
                y21 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = False) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=False)                    
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y12 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = True) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True)
                
                if not(tx1down is np.nan or tx1down >= largeTime) and (tx1up is np.nan or tx1up >= largeTime): # nua falls below num post deceleration
                    spect = y21*(self._tobs < self._tcross) + (w22a*y22+w12a*y12)/(w22a+w12a)*(self._tobs >= self._tcross)
                elif (tx1down is np.nan or tx1down >= largeTime) and not(tx1up is np.nan or tx1up >= largeTime): # nua grows above nucut post deceleration
                    spect = y21*(self._tobs < self._tcross) + (w22b*y22+w32b*y32)/(w22b+w32b)*(self._tobs >= self._tcross)
                else: # nua stays between num and nucut post deceleration
                    spect = y21*(self._tobs < self._tcross) + y22*(self._tobs >= self._tcross)
                                
            elif (self._numrs_tcross < self._nucutrs_tcross < self._nuars_tcross): # spectrum 3
                # Calculate tx1
                tx1 = self._tnuarseqnucutrsPreCrossISMcaseIIIc # (self._nucutrs_tcross/self._nuars_tcross)**(1/(laF1-lc1))*self._tcross                    
                
                if self._kGamma <= 1:
                    tx2 = self._tnuarseqnucutrsPostCrossISMcaseIc
                    tx3 = self._tnuarseqnumrsPostCrossISMcaseIc
                else:
                    tx2 = self._tnuarseqnucutrsPostCrossISMcaseIIc
                    tx3 = self._tnuarseqnumrsPostCrossISMcaseIIc
                
                w21 = 1./(1.+(self._tobs/tx1)**pl)                    
                w31 = 1./(1.+(self._tobs/tx1)**(-1.*pl)) 
                w32a = 1/(1 + (self._tobs/tx2)**pl) # a, and b are here because 2 potential crossings   
                w22a = 1/(1 + (self._tobs/tx2)**(-pl))
                w22b = 1/(1 + (self._tobs/tx3)**pl)
                w12b = 1/(1 + (self._tobs/tx3)**(-pl))
                   
                y21 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = False) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y31 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = False) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=True)   
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=True)   
                y12 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = True) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=True)   
                
                if tx2 is np.nan or tx2 >= largeTime:  
                    spect = (w21*y21 + w31*y31)/(w21+w31)*(self._tobs < self._tcross) + y32*(self._tobs >= self._tcross)
                elif tx3 is np.nan or tx3 >= largeTime: # in the case that post deceleration nua falls below nucut but not below num
                    spect = (w21*y21 + w31*y31)/(w21+w31)*(self._tobs < self._tcross) + (w32a*y32+w22a*y22)/(w32a+w22a)*(self._tobs >= self._tcross)
                else: # in the case that post deceleration nua falls below nucut and below num
                    spect = (w21*y21 + w31*y31)/(w21+w31)*(self._tobs < self._tcross) + (w32a*y32+(w22a*w22b)*y22+w12b*y12)/(w32a+w22a*w22b+w12b)*(self._tobs >= self._tcross)
            
            else: # TODO implement spectrum 4, 5, 6
                warnings.warn("fast cooling cases unimplemented")
                return np.full_like(self._nu, smallNum)
        
        else: # wind
            if (self._nuars_tcross < self._numrs_tcross < self._nucutrs_tcross):
                # Calculate tx2
                tx2 = self._tnuarseqnumrsPreCrossWindCaseIIIa # (self._nuars_tcross/self._numrs_tcross)**(1/(lm1-laD1))*tdec
                # Calculate tx1
                tx1 = self._tnuarseqnucutrsPreCrossWindCaseIIIa # (nua2/nuc2)**(1/(lc1-laE1))*tx2
                # Calculate tx3:
                if self._kGamma <= 1:
                    tx3 = self._tnuarseqnumrsPostCrossWindCaseIa # (self._numrs_tcross/self._nuars_tcross)**(1/(laD2-lm2))*tdec
                    tx4 = self._tnuarseqnucutrsPostCrossWindCaseIa # (nuc3/nua3)**(1/(laE2-lc2))*tx3
                else:
                    tx3 = self._tnuarseqnumrsPostCrossWindCaseIIa
                    tx4 = self._tnuarseqnucutrsPostCrossWindCaseIIa # (nuc3/nua3)**(1/(laE2-lc2))*tx3
                # Calculate tx4
                
                w31 = firstweight(self._tobs,tx1,10) # 1./(1.+(self._tobs/tx1)**10)                    
                w21 = (1./(1.+(self._tobs/tx1)**(-1.*10))) * (1./(1.+(self._tobs/tx2)**10))
                w11 = 1./(1.+(self._tobs/tx2)**(-1.*10))
                w12a = 1./(1.+(self._tobs/tx3)**1.*pl)                    
                w22a = 1./(1.+(self._tobs/tx3)**(-1.*pl))
                w22b = 1/(1+(self._tobs/tx4)**pl) # TODO check validity/if need to use firstweight function for all weights I've added
                w32b = 1/(1+(self._tobs/tx4)**(-pl))
                
                y31 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = False) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y21 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = False) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y11 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = False) # calc_spect(1, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y12 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = True) # calc_spect(1, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True)
                
                if tx3 is np.nan or tx3 >= largeTime: # if no second crossing then always in spectrum 1 post deceleration
                    spect = (w31*y31 + w21*y21 + w11*y11)/(w31+w21+w11)*(self._tobs < self._tcross) + y12*(self._tobs >= self._tcross)
                elif tx4 is np.nan or tx4 >= largeTime:    
                    spect = (w31*y31 + w21*y21 + w11*y11)/(w31+w21+w11)*(self._tobs < self._tcross) + (w12a*y12 + w22a*y22)/(w12a+w22a)*(self._tobs >= self._tcross)            
                else:
                    spect = (w31*y31 + w21*y21 + w11*y11)/(w31+w21+w11)*(self._tobs < self._tcross) + (w12a*y12 + (w22a*w22b)*y22 + w32b*y32)/(w12a+w22a*w22b+w32b)*(self._tobs >= self._tcross)
                
            elif (self._numrs_tcross < self._nuars_tcross < self._nucutrs_tcross):
                # Calculate tx1
                tx1 = self._tnuarseqnucutrsPreCrossWindCaseIIIb # (self._nuars_tcross/self._nucutrs_tcross)**(1/(lc1-laE1))*self._tcross
                # Calculate tx2
                if self._kGamma <=1:
                    tx2up = self._tnuarseqnucutrsPostCrossWindCaseIb # (self._nucutrs_tcross/self._nuars_tcross)**(1/(laE2-lc2))*self._tcross
                    tx2down = self._tnuarseqnumrsPostCrossWindCaseIb
                else:
                    tx2up = self._tnuarseqnucutrsPostCrossWindCaseIIb
                    tx2down = self._tnuarseqnumrsPostCrossWindCaseIIb
                
                w31 = 1./(1.+(self._tobs/tx1)**pl)                    
                w21 = 1./(1.+(self._tobs/tx1)**(-1.*pl))
                w22a = 1/(1+(self._tobs/tx2down)**pl) 
                w12a = 1/(1+(self._tobs/tx2down)**(-pl))
                w22b = 1/(1+(self._tobs/tx2up)**pl) 
                w32b = 1/(1+(self._tobs/tx2up)**(-pl))
                  
                y31 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = False) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y21 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = False) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=False)                    
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True) # calc_spect(2, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True)
                
                # spect = (w31*y31 + w21*y21)/(w31+w21)*(self._tobs < self._tcross) + y22*(self._tobs >= self._tcross)
                if not(tx2down is np.nan or tx2down >= largeTime) and (tx2up is np.nan or tx2up >= largeTime): # nua falls below num post deceleration
                    spect = (w31*y31 + w21*y21)/(w31+w21)*(self._tobs < self._tcross) + (w22a*y22+w12a*y12)/(w22a+w12a)*(self._tobs >= self._tcross)
                elif (tx2down is np.nan or tx2down >= largeTime) and not(tx2up is np.nan or tx2up >= largeTime): # nua grows above nucut post deceleration
                    spect = (w31*y31 + w21*y21)/(w31+w21)*(self._tobs < self._tcross) + (w22b*y22+w32b*y32)/(w22b+w32b)*(self._tobs >= self._tcross)
                else: # nua stays between num and nucut post deceleration
                    spect = (w31*y31 + w21*y21)/(w31+w21)*(self._tobs < self._tcross) + y22*(self._tobs >= self._tcross)
                                
                
            elif (self._numrs_tcross < self._nucutrs_tcross < self._nuars_tcross):
                # No crossings pre deceleration
                if self._kGamma <= 1:
                    tx3 = self._tnuarseqnumrsPostCrossWindCaseIc # (self._numrs_tcross/self._nuars_tcross)**(1/(laD2-lm2))*tdec
                    tx4 = self._tnuarseqnucutrsPostCrossWindCaseIc # (nuc3/nua3)**(1/(laE2-lc2))*tx3
                else:
                    tx3 = self._tnuarseqnumrsPostCrossWindCaseIIc
                    tx4 = self._tnuarseqnucutrsPostCrossWindCaseIIc # (nuc3/nua3)**(1/(laE2-lc2))*tx3
               

                w32a = 1/(1 + (self._tobs/tx3)**pl) # a, and b are here because 2 potential crossings   
                w22a = 1/(1 + (self._tobs/tx3)**(-pl))
                w22b = 1/(1 + (self._tobs/tx4)**pl)
                w12b = 1/(1 + (self._tobs/tx4)**(-pl))
               
                y31 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = False) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=False)
                y32 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 3, decelerated = True) # calc_spect(3, fsparams, f, nua, num, nuc, fnumax, decelerated=True)
                y22 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 2, decelerated = True)   
                y12 = RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, specnum = 1, decelerated = True) 
                
                #spect = y31*(self._tobs < self._tcross) + y32*(self._tobs >= self._tcross)
                if tx3 is np.nan or tx3 >= largeTime:  
                    spect = y31*(self._tobs < self._tcross) + y32*(self._tobs >= self._tcross)
                elif tx4 is np.nan or tx4 >= largeTime: # in the case that post deceleration nua falls below nucut but not below num
                    spect = y31*(self._tobs < self._tcross) + (w32a*y32+w22a*y22)/(w32a+w22a)*(self._tobs >= self._tcross)
                else: # in the case that post deceleration nua falls below nucut and below num
                    spect = y31*(self._tobs < self._tcross) + (w32a*y32+(w22a*w22b)*y22+w12b*y12)/(w32a+w22a*w22b+w12b)*(self._tobs >= self._tcross)
                
            else: # TODO implement spectrum 4, 5, 6
                warnings.warn("fast cooling cases unimplemented")
                return np.full_like(self._nu, smallNum)
        
        return spect
            
    @staticmethod
    def _spectrum(_tobs, _nu, _tcross, _Fnumaxrs, _numrs, _nucutrs, _nuars, _p, _k, diagnostic = False, specnum = None, decelerated = None):
        """Array-aware Granot-Sari spectrum dispatch.

        Was previously @np.vectorize, which Python-looped per (t, nu) point
        and dominated profiling. The replacement broadcasts the per-point
        inputs, classifies the spectral branch (1..5) per point, and runs
        one vectorised numpy pass per occupied branch via _spectrum_branch_v.
        """
        # Broadcast every "per-point" input to a common 1-D shape; preserve
        # the scalar-in / scalar-out contract the np.vectorize version had.
        scalar_inputs = all(np.ndim(x) == 0 for x in (_tobs, _nu, _Fnumaxrs, _numrs, _nucutrs, _nuars))
        tobs, nu, Fnumax, num, nucut, nuar = (
            np.atleast_1d(np.asarray(x, dtype=float))
            for x in (_tobs, _nu, _Fnumaxrs, _numrs, _nucutrs, _nuars)
        )
        n = max(tobs.size, nu.size, Fnumax.size, num.size, nucut.size, nuar.size)
        if tobs.size   < n: tobs   = np.broadcast_to(tobs,   (n,))
        if nu.size     < n: nu     = np.broadcast_to(nu,     (n,))
        if Fnumax.size < n: Fnumax = np.broadcast_to(Fnumax, (n,))
        if num.size    < n: num    = np.broadcast_to(num,    (n,))
        if nucut.size  < n: nucut  = np.broadcast_to(nucut,  (n,))
        if nuar.size   < n: nuar   = np.broadcast_to(nuar,   (n,))

        if diagnostic:
            # Match the upstream @np.vectorize behaviour: a 4-tuple of arrays
            # broadcast to the common shape. Scalars-in -> scalars-out.
            if scalar_inputs:
                return float(nuar[0]), float(num[0]), float(nucut[0]), float(Fnumax[0])
            return nuar, num, nucut, Fnumax

        if decelerated is None:
            cut = tobs > _tcross
        else:
            cut = np.full(n, bool(decelerated))

        if specnum is None:
            branch = _classify_branch_v(smallNum, nuar, num, nucut)
        else:
            branch = np.full(n, int(specnum), dtype=np.int8)

        Fnu_true = _obs_flux_max_v(Fnumax, nuar, num, nucut, _p, branch,
                                   specnum_forced=specnum)

        out = np.zeros(n, dtype=float)
        for sn in (1, 2, 3, 4, 5):
            mask = branch == sn
            if not mask.any():
                continue
            out[mask] = _spectrum_branch_v(
                sn, nu[mask], Fnu_true[mask], smallNum,
                nuar[mask], num[mask], nucut[mask],
                _p, _k, cut[mask],
            )

        if scalar_inputs:
            return float(out[0])
        return out
        
    @np.vectorize
    def _ISM(k):
        """"""
        if k==0:
            return True
        elif k==2:
            return False
        else:
            raise Exception("k must be 0 (ISM) or 2 (wind)")
    
    @np.vectorize
    def _g(ISM, g):
        """"""
        if g is None:
            if ISM:
                return 2
            else:
                return 1
        else:
            return g
    
    def _compute_a(self):
        """"""
        RSjetStruct._warn_a(self._keps)
        
        return np.where(self._keps < 2, self._keps, 2)
    
    @np.vectorize
    def _warn_a(keps):
        """warns for keps > 2 since scalings assume keps < 2.
        """
        if keps > 2:
            warnings.warn("scalings assume keps < 2", RuntimeWarning)
    
    def _compute_A(self):
        """"""
        return RSjetStruct._COMPUTE_A(self._a, self._kGamma)
    
    @np.vectorize
    def _COMPUTE_A(a, kGamma):
        """"""
        if kGamma == 0: # TODO what should this return if kGamma is zero?
            return 1/smallNum
        else:
            return a/kGamma
    
    def _cases(self, observable, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII,\
               windScale_caseI, windScale_caseII, windScale_caseIII):
        """"""
        caseISM_IandII = self._caseIorII(ISMscale_caseI * observable, ISMscale_caseII * observable)
            
        caseWind_IandII = self._caseIorII(windScale_caseI * observable, windScale_caseII * observable)
    
        return self._caseISMorWind(self._casePreOrPost(ISMscale_caseIII * observable,\
                                                       caseISM_IandII),\
                                   self._casePreOrPost(windScale_caseIII * observable,\
                                                       caseWind_IandII))
            
    def _casesabc(self, observable, ISMscale_caseIa, ISMscale_caseIb, ISMscale_caseIc, ISMscale_caseIIa, ISMscale_caseIIb, ISMscale_caseIIc,\
                  windScale_caseIa, windScale_caseIb, windScale_caseIc, windScale_caseIIa, windScale_caseIIb, windScale_caseIIc):
        """"""
        if self._ISM:
            nuars_ISMcaseIa   = self._compute_caseA(observable, ISMscale_caseIa,   self._tnuarseqnumrsPostCrossISMcaseIa,   self._tnuarseqnucutrsPostCrossISMcaseIa,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIa,  self._tnuarseqnucutrsPreCrossISMcaseIIIa,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
            nuars_ISMcaseIb   = self._compute_caseB(observable, ISMscale_caseIb,   self._tnuarseqnumrsPostCrossISMcaseIb,   self._tnuarseqnucutrsPostCrossISMcaseIb,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIb,  self._tnuarseqnucutrsPreCrossISMcaseIIIb,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
            nuars_ISMcaseIc   = self._compute_caseC(observable, ISMscale_caseIc,   self._tnuarseqnumrsPostCrossISMcaseIc,   self._tnuarseqnucutrsPostCrossISMcaseIc,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIc,  self._tnuarseqnucutrsPreCrossISMcaseIIIc,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
            nuars_ISMcaseIIa  = self._compute_caseA(observable, ISMscale_caseIIa,  self._tnuarseqnumrsPostCrossISMcaseIIa,  self._tnuarseqnucutrsPostCrossISMcaseIIa,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIa,  self._tnuarseqnucutrsPreCrossISMcaseIIIa,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
            nuars_ISMcaseIIb  = self._compute_caseB(observable, ISMscale_caseIIb,  self._tnuarseqnumrsPostCrossISMcaseIIb,  self._tnuarseqnucutrsPostCrossISMcaseIIb,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIb,  self._tnuarseqnucutrsPreCrossISMcaseIIIb,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
            nuars_ISMcaseIIc  = self._compute_caseC(observable, ISMscale_caseIIc,  self._tnuarseqnumrsPostCrossISMcaseIIc,  self._tnuarseqnucutrsPostCrossISMcaseIIc,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIc,  self._tnuarseqnucutrsPreCrossISMcaseIIIc,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        else:
            nuars_ISMcaseIa   = np.full_like(self._tobs, smallNum)
            nuars_ISMcaseIb   = np.full_like(self._tobs, smallNum)
            nuars_ISMcaseIc   = np.full_like(self._tobs, smallNum)
            nuars_ISMcaseIIa  = np.full_like(self._tobs, smallNum)
            nuars_ISMcaseIIb  = np.full_like(self._tobs, smallNum)
            nuars_ISMcaseIIc  = np.full_like(self._tobs, smallNum)

        if not(self._ISM):
            nuars_windCaseIa  = self._compute_caseA(observable, windScale_caseIa,  self._tnuarseqnumrsPostCrossWindCaseIa,  self._tnuarseqnucutrsPostCrossWindCaseIa,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIa, self._tnuarseqnucutrsPreCrossWindCaseIIIa, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
            nuars_windCaseIb  = self._compute_caseB(observable, windScale_caseIb,  self._tnuarseqnumrsPostCrossWindCaseIb,  self._tnuarseqnucutrsPostCrossWindCaseIb,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIb, self._tnuarseqnucutrsPreCrossWindCaseIIIb, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
            nuars_windCaseIc  = self._compute_caseC(observable, windScale_caseIc,  self._tnuarseqnumrsPostCrossWindCaseIc,  self._tnuarseqnucutrsPostCrossWindCaseIc,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIc, self._tnuarseqnucutrsPreCrossWindCaseIIIc, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
            nuars_windCaseIIa = self._compute_caseA(observable, windScale_caseIIa, self._tnuarseqnumrsPostCrossWindCaseIIa, self._tnuarseqnucutrsPostCrossWindCaseIIa, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIa, self._tnuarseqnucutrsPreCrossWindCaseIIIa, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
            nuars_windCaseIIb = self._compute_caseB(observable, windScale_caseIIb, self._tnuarseqnumrsPostCrossWindCaseIIb, self._tnuarseqnucutrsPostCrossWindCaseIIb, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIb, self._tnuarseqnucutrsPreCrossWindCaseIIIb, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
            nuars_windCaseIIc = self._compute_caseC(observable, windScale_caseIIc, self._tnuarseqnumrsPostCrossWindCaseIIc, self._tnuarseqnucutrsPostCrossWindCaseIIc, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIc, self._tnuarseqnucutrsPreCrossWindCaseIIIc, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        else:
            nuars_windCaseIa  = np.full_like(self._tobs, smallNum)
            nuars_windCaseIb  = np.full_like(self._tobs, smallNum)
            nuars_windCaseIc  = np.full_like(self._tobs, smallNum)
            nuars_windCaseIIa = np.full_like(self._tobs, smallNum)
            nuars_windCaseIIb = np.full_like(self._tobs, smallNum)
            nuars_windCaseIIc = np.full_like(self._tobs, smallNum)
        
        return self._caseISMorWind(self._caseIorII(self._caseAorBorC(nuars_ISMcaseIa,\
                                                                     nuars_ISMcaseIb,\
                                                                     nuars_ISMcaseIc),\
                                                   self._caseAorBorC(nuars_ISMcaseIIa,\
                                                                     nuars_ISMcaseIIb,\
                                                                     nuars_ISMcaseIIc)),\
                                   self._caseIorII(self._caseAorBorC(nuars_windCaseIa,\
                                                                     nuars_windCaseIb,\
                                                                     nuars_windCaseIc),\
                                                   self._caseAorBorC(nuars_windCaseIIa,\
                                                                     nuars_windCaseIIb,\
                                                                     nuars_windCaseIIc)))
    
    def _caseISMorWind(self, i1, i2):
        """"""
        if self._ISM:
            return i1
        else:
            return i2
    
    def _casePreOrPost(self, i1, i2):
        """"""
        return np.where(self._tobs < self._tcross, i1, i2)
        
    def _caseIorII(self, i1, i2):
        """"""
        return np.where(self._kGamma <= 1, i1, i2)
    
    def _caseAorBorC(self, i1, i2, i3): # TODO orC
        """"""
        return np.where((self._nuars_tcross < self._numrs_tcross) & (self._numrs_tcross < self._nucutrs_tcross), i1,\
                        np.where((self._numrs_tcross < self._nuars_tcross) & (self._nuars_tcross < self._nucutrs_tcross), i2,\
                                 np.where((self._numrs_tcross < self._nucutrs_tcross) & (self._nucutrs_tcross < self._nuars_tcross), i3,\
                                           smallNum * 2))) # FIXME
    
    def _BCmerge(self, nuars_caseX, caseBstr, caseCstr, above = True):
        """"""
        caseStr = caseBstr[:-1]
        
        merge = (self._alphaDict["nuars"][caseCstr] < self._alphaDict["nucutrs"][caseStr]) &\
                (self._alphaDict["nuars"][caseBstr] > self._alphaDict["nucutrs"][caseStr])
        
        if above:
            order = nuars_caseX < self._nucutrs
        else:
            order = nuars_caseX > self._nucutrs
        
        return np.where(order & merge, self._nucutrs, nuars_caseX)

    def _compute_postCrossCaseA(self, observable, scale_caseA, teqmcaseA, teqcutcaseA, caseAstr, caseBstr, caseCstr):
        """"""
        nuars_caseA = scale_caseA * observable
        nuars_caseA =    where(nuars_caseA < self._numrs, lambda x: x,\
                                                          lambda t: observable *\
                                                          (teqmcaseA/self._tcross)**self._alphaDict["nuars"][caseAstr] *\
                                                          (t/teqmcaseA)**self._alphaDict["nuars"][caseBstr], argsa = [nuars_caseA], argsb = [self._tobs])
        nuars_caseA = self._BCmerge(nuars_caseA, caseBstr, caseCstr, above = False)
        
        if np.any(np.logical_not(nuars_caseA <= self._nucutrs)):
            nuars_caseA =   where((nuars_caseA <= self._nucutrs), lambda x: x,\
                                                                  lambda t: observable *\
                                                                  (teqmcaseA/self._tcross)**self._alphaDict["nuars"][caseAstr] *\
                                                                  (teqcutcaseA/teqmcaseA)**self._alphaDict["nuars"][caseBstr] *\
                                                                  (t/teqcutcaseA)**self._alphaDict["nuars"][caseCstr], argsa = [nuars_caseA], argsb = [self._tobs])
        
        return nuars_caseA
    
    def _compute_preCrossCaseA(self, observable, teqmcaseIIIA, teqcutcaseIIIA, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        nuars_caseA = self._tfrac**self._alphaDict["nuars"][caseIIIAstr] * observable
        nuars_caseA = np.where(nuars_caseA < self._numrs, nuars_caseA,\
                                                          observable *\
                                                          (teqmcaseIIIA/self._tcross)**self._alphaDict["nuars"][caseIIIAstr]*\
                                                          (self._tobs/teqmcaseIIIA)**self._alphaDict["nuars"][caseIIIBstr])
        nuars_caseA = np.where(nuars_caseA < self._nucutrs, nuars_caseA,\
                                                            observable *\
                                                            (teqmcaseIIIA/self._tcross)**self._alphaDict["nuars"][caseIIIAstr] *\
                                                            (teqcutcaseIIIA/teqmcaseIIIA)**self._alphaDict["nuars"][caseIIIBstr] *\
                                                            (self._tobs/teqcutcaseIIIA)**self._alphaDict["nuars"][caseIIICstr])
        
        return nuars_caseA
            
    def _compute_caseA(self, observable, scale_caseA, teqmcaseA, teqcutcaseA, caseAstr, caseBstr, caseCstr,\
                       teqmcaseIIIA, teqcutcaseIIIA, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        postCross = self._compute_postCrossCaseA(observable, scale_caseA, teqmcaseA, teqcutcaseA, caseAstr, caseBstr, caseCstr)
        preCross = self._compute_preCrossCaseA(observable, teqmcaseIIIA, teqcutcaseIIIA, caseIIIAstr, caseIIIBstr, caseIIICstr)
        
        return np.where(self._tobs < self._tcross, preCross, postCross)
    
    def _compute_postCrossCaseB(self, observable, scale_caseB, teqmcaseB, teqcutcaseB, caseAstr, caseBstr, caseCstr):
        """"""
        nuars_caseB = scale_caseB * observable
        nuars_caseB =    where(nuars_caseB > self._numrs, lambda x: x,\
                                                          lambda t: observable *\
                                                          (teqmcaseB/self._tcross)**self._alphaDict["nuars"][caseBstr] *\
                                                          (t/teqmcaseB)**self._alphaDict["nuars"][caseAstr], argsa = [nuars_caseB], argsb = [self._tobs])
        nuars_caseB = self._BCmerge(nuars_caseB, caseBstr, caseCstr, above = False)
        nuars_caseB =    where((nuars_caseB <= self._nucutrs), lambda x: x,\
                                                              lambda t: observable *\
                                                              (teqcutcaseB/self._tcross)**self._alphaDict["nuars"][caseBstr] *\
                                                              (t/teqcutcaseB)**self._alphaDict["nuars"][caseCstr], argsa = [nuars_caseB], argsb = [self._tobs])
    
        return nuars_caseB
    
    def _compute_preCrossCaseB(self, observable, teqmcaseIIIB, teqcutcaseIIIB, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        nuars_caseB = self._tfrac**(self._alphaDict["nuars"][caseIIIBstr]) * observable
        nuars_caseB = np.where(nuars_caseB > self._numrs, nuars_caseB,\
                                                          observable *\
                                                          (teqmcaseIIIB/self._tcross)**self._alphaDict["nuars"][caseIIIBstr] *\
                                                          (self._tobs/teqmcaseIIIB)**self._alphaDict["nuars"][caseIIIAstr])
        nuars_caseB = np.where(nuars_caseB < self._nucutrs, nuars_caseB,\
                                                            observable *\
                                                            (teqcutcaseIIIB/self._tcross)**self._alphaDict["nuars"][caseIIIBstr] *\
                                                            (self._tobs/teqcutcaseIIIB)**self._alphaDict["nuars"][caseIIICstr])
        return nuars_caseB
    
    def _compute_caseB(self, observable, scale_caseB, teqmcaseB, teqcutcaseB, caseAstr, caseBstr, caseCstr,\
                       teqmcaseIIIB, teqcutcaseIIIB, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        postCross = self._compute_postCrossCaseB(observable, scale_caseB, teqmcaseB, teqcutcaseB, caseAstr, caseBstr, caseCstr)
        preCross = self._compute_preCrossCaseB(observable, teqmcaseIIIB, teqcutcaseIIIB, caseIIIAstr, caseIIIBstr, caseIIICstr)
        
        return np.where(self._tobs < self._tcross, preCross, postCross)
    
    def _compute_postCrossCaseC(self, observable, scale_caseC, teqmcaseC, teqcutcaseC, caseAstr, caseBstr, caseCstr): # TODO
        """"""
        
        nuars_caseC = scale_caseC * observable
        nuars_caseC = self._BCmerge(nuars_caseC, caseBstr, caseCstr, above = True)
        nuars_caseC =    where(nuars_caseC >= self._nucutrs, lambda x: x,\
                                                             lambda t: observable *\
                                                             (teqcutcaseC/self._tcross)**self._alphaDict["nuars"][caseCstr] *\
                                                             (t/teqcutcaseC)**self._alphaDict["nuars"][caseBstr], argsa = [nuars_caseC], argsb = [self._tobs])
        if np.any(np.logical_not(nuars_caseC > self._numrs)):
            nuars_caseC =    where(nuars_caseC > self._numrs, lambda x: x,\
                                                              lambda t: observable *\
                                                              (teqcutcaseC/self._tcross)**self._alphaDict["nuars"][caseCstr] *\
                                                              (teqmcaseC/teqcutcaseC)**self._alphaDict["nuars"][caseBstr] *\
                                                              (t/teqmcaseC)**self._alphaDict["nuars"][caseAstr], argsa = [nuars_caseC], argsb = [self._tobs])
            
        return nuars_caseC
        
    def _compute_preCrossCaseC(self, observable, teqmcaseIIIC, teqcutcaseIIIC, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        nuars_caseC = self._tfrac**self._alphaDict["nuars"][caseIIICstr] * observable
        nuars_caseC = np.where(nuars_caseC > self._nucutrs, nuars_caseC,\
                                                            observable *\
                                                            (teqcutcaseIIIC/self._tcross)**self._alphaDict["nuars"][caseIIICstr] *\
                                                            (self._tobs/teqcutcaseIIIC)**self._alphaDict["nuars"][caseIIIBstr])
        nuars_caseC = np.where(nuars_caseC > self._numrs, nuars_caseC,\
                                                          observable *\
                                                          (teqcutcaseIIIC/self._tcross)**self._alphaDict["nuars"][caseIIICstr] *\
                                                          (teqmcaseIIIC/teqcutcaseIIIC)**self._alphaDict["nuars"][caseIIIBstr] *\
                                                          (self._tobs/teqmcaseIIIC)**self._alphaDict["nuars"][caseIIICstr])
        
        return nuars_caseC
            
    def _compute_caseC(self, observable, scale_caseC, teqmcaseC, teqcutcaseC, caseAstr, caseBstr, caseCstr,\
                       teqmcaseIIIC, teqcutcaseIIIC, caseIIIAstr, caseIIIBstr, caseIIICstr):
        """"""
        postCross = self._compute_postCrossCaseC(observable, scale_caseC, teqmcaseC, teqcutcaseC, caseAstr, caseBstr, caseCstr)
        preCross = self._compute_preCrossCaseC(observable, teqmcaseIIIC, teqcutcaseIIIC, caseIIIAstr, caseIIIBstr, caseIIICstr)
        
        return np.where(self._tobs < self._tcross, preCross, postCross)
    
    def _compute_casePostJet(self, observable, ISMscale_caseIV, ISMscale_caseV, windScale_caseIV, windScale_caseV):
        """"""
        return np.where(self._ISM, np.where(self._kGamma <= 1, observable * ISMscale_caseIV,\
                                                              observable * ISMscale_caseV),\
                                   np.where(self._kGamma <= 1, observable * windScale_caseIV,\
                                                              observable * windScale_caseV))
    
    def _compute_casePostNRFS(self, observable, ISMscale_caseVI, ISMscale_caseVII, windScale_caseVI, windScale_caseVII):
        """"""
        return np.where(self._ISM, np.where(self._kGamma <= 1, observable * ISMscale_caseVI,\
                                                              observable * ISMscale_caseVII),\
                                   np.where(self._kGamma <= 1, observable * windScale_caseVI,\
                                                              observable * windScale_caseVII))
    
    def _caseJet(self, i1, i2):
        """"""
        return np.where(self._tobs < self._tjet, i1, i2)
    
    def _caseNRFS(self, i1, i2):
        return np.where(self._tobs < self._tNRFS, i1, i2)
    
    def Fnumaxrs(self):
        """"""
        if self._ISM:
            ISMscale_caseI = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseI"])
            ISMscale_caseII = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseII"])
            ISMscale_caseIII = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseIII"])
        else:
            ISMscale_caseI = np.full_like(self._tobs, smallNum)
            ISMscale_caseII = np.full_like(self._tobs, smallNum)
            ISMscale_caseIII = np.full_like(self._tobs, smallNum)
            
        if not(self._ISM):
            windScale_caseI = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseI"])
            windScale_caseII = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseII"])
            windScale_caseIII = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseIII"])
        else:
            windScale_caseI = np.full_like(self._tobs, smallNum)
            windScale_caseII = np.full_like(self._tobs, smallNum)
            windScale_caseIII = np.full_like(self._tobs, smallNum)
            
        
        with np.errstate(divide = "raise"):
            try:
                if self._ISM:
                    ISMscale_caseIV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseI"]) *\
                                      (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseIV"])
                    ISMscale_caseV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseII"]) *\
                                     (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseV"])
                                      
                    ISMscale_caseVI = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseI"]) *\
                                      (self._tNRFS/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseIV"]) *\
                                      (self._tobs/self._tNRFS)**(self._alphaDict["Fnumaxrs"]["ISMcaseVI"])                                      
                    ISMscale_caseVII = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseII"]) *\
                                     (self._tNRFS/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseV"]) *\
                                     (self._tobs/self._tNRFS)**(self._alphaDict["Fnumaxrs"]["ISMcaseVII"])
                else:
                    ISMscale_caseIV = np.full_like(self._tobs, smallNum)
                    ISMscale_caseV = np.full_like(self._tobs, smallNum)
                    ISMscale_caseVI = np.full_like(self._tobs, smallNum)                 
                    ISMscale_caseVII = np.full_like(self._tobs, smallNum)
                    
                if not(self._ISM):
                    windScale_caseIV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseI"]) *\
                                       (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseIV"])
                    windScale_caseV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseII"]) *\
                                      (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseV"])
                    
                    windScale_caseVI = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseI"]) *\
                                       (self._tNRFS/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseIV"]) *\
                                       (self._tobs/self._tNRFS)**(self._alphaDict["Fnumaxrs"]["windCaseVI"]) 
                    windScale_caseVII = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseII"]) *\
                                      (self._tNRFS/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseV"]) *\
                                      (self._tobs/self._tNRFS)**(self._alphaDict["Fnumaxrs"]["windCaseVII"]) 
                else:
                    windScale_caseIV = np.full_like(self._tobs, smallNum)
                    windScale_caseV = np.full_like(self._tobs, smallNum)
                    windScale_caseVI = np.full_like(self._tobs, smallNum)
                    windScale_caseVII = np.full_like(self._tobs, smallNum)
                                  
            except (OverflowError, ZeroDivisionError, FloatingPointError):
                ISMscale_caseIV = np.nan
                ISMscale_caseV = np.nan
                windScale_caseIV = np.nan
                windScale_caseV = np.nan
                
                ISMscale_caseVI = np.nan
                ISMscale_caseVII = np.nan
                windScale_caseVI = np.nan
                windScale_caseVII = np.nan
                
    
        Fnumaxrs_preJet = self._cases(self._Fnumaxrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        Fnumaxrs_postJet = self._compute_casePostJet(self._Fnumaxrs_tcross, ISMscale_caseIV, ISMscale_caseV, windScale_caseIV, windScale_caseV)
        Fnumaxrs_postNRFS = self._compute_casePostNRFS(self._Fnumaxrs_tcross, ISMscale_caseVI, ISMscale_caseVII, windScale_caseVI, windScale_caseVII)
        
        return self._caseNRFS(self._caseJet(Fnumaxrs_preJet, Fnumaxrs_postJet), Fnumaxrs_postNRFS)
        
    def numrs(self):
        """"""
        if self._ISM:
            ISMscale_caseI = self._tfrac**(self._alphaDict["numrs"]["ISMcaseI"])
            ISMscale_caseII = self._tfrac**(self._alphaDict["numrs"]["ISMcaseII"])
            ISMscale_caseIII = self._tfrac**(self._alphaDict["numrs"]["ISMcaseIII"])
        else:
            ISMscale_caseI = np.full_like(self._tobs, smallNum)
            ISMscale_caseII = np.full_like(self._tobs, smallNum)
            ISMscale_caseIII = np.full_like(self._tobs, smallNum)

        if not(self._ISM):
            windScale_caseI = self._tfrac**(self._alphaDict["numrs"]["windCaseI"])
            windScale_caseII = self._tfrac**(self._alphaDict["numrs"]["windCaseII"])
            windScale_caseIII = self._tfrac**(self._alphaDict["numrs"]["windCaseIII"])
        else:
            windScale_caseI = np.full_like(self._tobs, smallNum)
            windScale_caseII = np.full_like(self._tobs, smallNum)
            windScale_caseIII = np.full_like(self._tobs, smallNum)
            
        
        return self._cases(self._numrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        
    def nucutrs(self):
        """"""
    
        if self._ISM:
            ISMscale_caseI = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseI"])
            ISMscale_caseII = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseII"])
            ISMscale_caseIII = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseIII"])
        else:
            ISMscale_caseI = np.full_like(self._tobs, smallNum)
            ISMscale_caseII = np.full_like(self._tobs, smallNum)
            ISMscale_caseIII = np.full_like(self._tobs, smallNum)

        if not(self._ISM):
            windScale_caseI = self._tfrac**(self._alphaDict["nucutrs"]["windCaseI"])
            windScale_caseII = self._tfrac**(self._alphaDict["nucutrs"]["windCaseII"])
            windScale_caseIII = self._tfrac**(self._alphaDict["nucutrs"]["windCaseIII"])
        else:
            windScale_caseI = np.full_like(self._tobs, smallNum)
            windScale_caseII = np.full_like(self._tobs, smallNum)
            windScale_caseIII = np.full_like(self._tobs, smallNum)
        
        return self._cases(self._nucutrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        
    def nuars(self):
        """"""
        if self._ISM:
            ISMscale_caseIa = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIa"])
            ISMscale_caseIb = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIb"])
            ISMscale_caseIc = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIc"])
            ISMscale_caseIIa = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIa"])
            ISMscale_caseIIb = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIb"])
            ISMscale_caseIIc = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIc"])
        else:
            ISMscale_caseIa = np.full_like(self._tobs, smallNum)
            ISMscale_caseIb = np.full_like(self._tobs, smallNum)
            ISMscale_caseIc = np.full_like(self._tobs, smallNum)
            ISMscale_caseIIa = np.full_like(self._tobs, smallNum)
            ISMscale_caseIIb = np.full_like(self._tobs, smallNum)
            ISMscale_caseIIc = np.full_like(self._tobs, smallNum)

        if not(self._ISM):
            windScale_caseIa = self._tfrac**(self._alphaDict["nuars"]["windCaseIa"])
            windScale_caseIb = self._tfrac**(self._alphaDict["nuars"]["windCaseIb"])
            windScale_caseIc = self._tfrac**(self._alphaDict["nuars"]["windCaseIc"])
            windScale_caseIIa = self._tfrac**(self._alphaDict["nuars"]["windCaseIIa"])
            windScale_caseIIb = self._tfrac**(self._alphaDict["nuars"]["windCaseIIb"])
            windScale_caseIIc = self._tfrac**(self._alphaDict["nuars"]["windCaseIIc"])
        else:
            windScale_caseIa = np.full_like(self._tobs, smallNum)
            windScale_caseIb = np.full_like(self._tobs, smallNum)
            windScale_caseIc = np.full_like(self._tobs, smallNum)
            windScale_caseIIa = np.full_like(self._tobs, smallNum)
            windScale_caseIIb = np.full_like(self._tobs, smallNum)
            windScale_caseIIc = np.full_like(self._tobs, smallNum)
        
        return self._casesabc(self._nuars_tcross, ISMscale_caseIa, ISMscale_caseIb, ISMscale_caseIc, ISMscale_caseIIa, ISMscale_caseIIb, ISMscale_caseIIc,\
                              windScale_caseIa, windScale_caseIb, windScale_caseIc, windScale_caseIIa, windScale_caseIIb, windScale_caseIIc)
    
    #def Fnumaxrsobs(self):
    #    """Computes the observed maximum Flux of the reverse shock (applying 
    #    synchrotron self absorption).
    #    """
    #    return obsFluxMax(self.Fnumaxrs(), 0, self._nuars, self._numrs, self._nucutrs, self._p)
      
    def _buildGamma3alphaDict(self):
        """
        Note : does not assume keps < 2
        """
        
        krat = RSjetStruct._KRAT(self._keps, self._kGamma)
        
        d = {
            "ISMcaseI"    : -3 * self._g/(3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._keps * self._g),
            "ISMcaseII"   : -3/(8 - krat),
            "windCaseI"   : -self._g/(1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._keps * self._g),
            "windCaseII"  : -1/(4 - krat),
            
            "ISMcaseIII"  : 0, # Kobayashi 2000 (5)
            "windCaseIII" : np.nan # TODO
        }
        
        return d

    @np.vectorize
    def _KRAT(keps, kGamma):
        """"""
        if kGamma == 0:
            return 1/smallNum
        else:
            return keps/kGamma
        
    def _buildAlphaDict(self): # TODO general 3 for ISM case Ic and IIc
        """Case I is for k_Gamma <= 1 and case II is for k_Gamma > 1 both for 
        time between t_cross and t_jet. Case III is for time less than t_cross.
        Case IV is post jet break for k_Gamma <= 1 and case V is for k_Gamma > 1 both for 
        time greater than t_jet. Case VI is post nonrelativistic forward shock 
        for k_Gamma <= 1 and case VII is for k_Gamma > 1. Case a is 
        nu_a < nu_m < nu_c, case b is nu_m < nu_a < nu_c, case c is 
        nu_m < nu_c < nu_a, case d is nu_a < nu_c < nu_m, case e is 
        nu_c < n_a < nu_m, case f is nu_c < n_m < n_a.
        
        Note : assumes keps < 2
        """
        
        d = {
        "Fnumaxrs" :
            {
            "ISMcaseI"     : -3 * (12 + 11 * self._g - 4 * self._kGamma * (3 + self._g) + 7 * self._a * self._g)/(7 * (3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._a * self._g)), # -6 * (17 - 10 * self._kGamma + 7 * self._a)/(7 * (15 + self._kGamma - 2 * self._a)), # ZWZ24 (43)
            "ISMcaseII"    : -(3 * self._A +3)/(8 - self._A), # ZWZ24 (54)
            "windCaseI"    : - (12 + 11 * self._g - 2 * self._kGamma * (6 - 5 * self._g))/(7 * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # -(23 - 2 * self._kGamma)/(7 * (3 + self._kGamma - self._a)), # ZWZ24 (67)
            "windCaseII"   : -3/(4 - self._A), # ZWZ24 (78)
            
            "ISMcaseIII"   : 3/2,
            "windCaseIII"  : -1/2
            },
        "numrs":
            {
            "ISMcaseI"     : -3 * (3 * (8 + 5 * self._g) - self._kGamma * (24 + self._g))/(7 * (3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._a * self._g)), # -6 * (27 - 13 * self._kGamma)/(7 * (15 + self._kGamma - 2 * self._a)), # ZWZ24 (44)
            "ISMcaseII"    : -6/(8 - self._A), # ZWZ24 (55)
            "windCaseI"    : -(3 * (8 + 5 * self._g) - self._kGamma * (24 - 13 * self._g) - 7 * self._a * self._g)/(7 * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # -(39 - 11 * self._kGamma - 7 * self._a)/(7 * (3 + self._kGamma - self._a)), # ZWZ24 (68)
            "windCaseII"   : -1, # ZWZ24 (79)
            
            "ISMcaseIII"   : 6,
            "windCaseIII"  : 1
            },
        "nucutrs":
            {
            "ISMcaseI"     : -(9 * (8 + 5 * self._g) - self._kGamma * (72 + 17 * self._g) - 14 * self._a * self._g)/(7 * (3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._a * self._g)), # -2 * (81 - 53 * self._kGamma - 14 * self._a)/(7 * (15 + self._kGamma - 2 * self._a)),  # ZWZ24 (45)
            "ISMcaseII"    : (2 * self._A - 4)/(8 - self._A), # ZWZ24 (56)
            "windCaseI"    : -(3 * (8 + 5 * self._g) - self._kGamma * (24 + 29 * self._g) + 7 * self._a * self._g)/(7 * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # -(39 - 53 * self._kGamma + 7 * self._a)/(7 * (3 + self._kGamma - self._a)), # ZWZ24 (69)
            "windCaseII"   : (2 - self._A)/(4 - self._A), # ZWZ24 (80)
            
            "ISMcaseIII"   : -2,
            "windCaseIII"  : 1
            },
        "nuars":
            {
            "ISMcaseIa"    : -3 * (3 * (12 + 11 * self._g) - self._kGamma * (36 - 23 * self._g) + 7 * self._a * self._g)/(35 * (3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._a * self._g)), # -3 * (14 * self._a + 10 * self._kGamma + 102)/(35 * (15 + self._kGamma - 2 * self._a)), # ZWZ24 (46)
            "ISMcaseIb"    : -(3 * (3 * self._p * (8 + 5 * self._g) + 8 * (5 + 4 * self._g)) - self._kGamma * (3 * self._p * (24 + self._g) + 4 * (30 - 11 * self._g)) + 14 * self._a * self._g)/(7 * (self._p + 4) * (3 * (1 + 2 * self._g) - self._kGamma * (3 - 2 * self._g) - self._a * self._g)), # -(28 * self._a - 78 * self._p * self._kGamma - 32 * self._kGamma + 162 * self._p + 312)/(7 * (self._p + 4) * (15 + self._kGamma - 2 * self._a)), # ZWZ24 (47)
            "ISMcaseIc"    : -3 * (8 + 5 * self._g)/(7 * (1 + 2 * self._g)), # TODO currently for non-structured jet add structured jet later
            "ISMcaseIIa"   : -3 * (8 + self._A)/(5 * (8 - self._A)), # ZWZ24 (57)
            "ISMcaseIIb"   : -2 * (self._A + 3 * self._p + 10)/((self._p + 4) * (8 - self._A)), # ZWZ24 (58)
            "ISMcaseIIc"   : -3 * (8 + 5 * self._g)/(7 * (1 + 2 * self._g)), # TODO currently for non-structured jet add structured jet later
            
            "windCaseIa"   : -(3 * (12 + 11 * self._g) - self._kGamma * (36 - 107 * self._g) - 35 * self._a * self._g)/(35 * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # -(69 + 71 * self._kGamma - 35 * self._a)/(35 * (3 + self._kGamma - self._a)), # ZWZ24 (71)
            "windCaseIb"   : -(3 * self._p * (8 + 5 * self._g) + 8 * (5 + 4 * self._g) - self._kGamma * (self._p * (24 - 13 * self._g) + 40 * (1 - 2 * self._g)) - 7 * (self._p + 4) * self._a * self._g)/(7 * (self._p + 4) * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # -((39 - 11 * self._kGamma - 7 * self._a) * self._p + 40 * self._kGamma - 28 * self._a + 72)/(7 * (self._p + 4) * (3 + self._kGamma - self._a)), # ZWZ24 (70)
            "windCaseIc"   : -(45 * (8 + 5 * self._g) - self._kGamma * (360 - 111 * self._g) - 77 * self._a * self._g)/(105 * (1 + 2 * self._g - self._kGamma * (1 - 2 * self._g) - self._a * self._g)), # derived from Zou, Wou, and Dai 2005 (ZWD05) (38) and Table 1
            "windCaseIIa"  : -1, # ZWZ24 (82)
            "windCaseIIb"  : -1, # ZWZ24 (81)
            "windCaseIIc"  : -(48 - 11 * self._A)/(15 * (4 - self._A)), # derived from ZWD05 (38) an Table 1
            
            "ISMcaseIIIa"  : -33/10,
            "ISMcaseIIIb"  : (6 * self._p - 7)/(self._p + 4),
            "ISMcaseIIIc"  : (6 * self._p - 9)/(self._p + 5),
            "ISMcaseIIId"  : 7/10,
            "ISMcaseIIIe"  : -1/2,
            "ISMcaseIIIf"  : (6 * self._p - 9)/(self._p + 5),
            
            "windCaseIIIa" : -23/10,
            "windCaseIIIb" : (self._p - 7)/(self._p + 4),
            "windCaseIIIc" : (self._p - 6)/(self._p + 5),
            "windCaseIIId" : -23/10,
            "windCaseIIIe" : -5/6,
            "windCaseIIIf" : (self._p - 6)/(self._p + 5)
            }
        }
        
        # jet break
        d["Fnumaxrs"]["ISMcaseIV"]  = d["Fnumaxrs"]["ISMcaseI"]   + self._Gamma3alphaDict["ISMcaseI"] * 2
        d["Fnumaxrs"]["ISMcaseV"]   = d["Fnumaxrs"]["ISMcaseII"]  + self._Gamma3alphaDict["ISMcaseII"] * 2
        d["Fnumaxrs"]["windCaseIV"] = d["Fnumaxrs"]["windCaseI"]  + self._Gamma3alphaDict["windCaseI"] * 2
        d["Fnumaxrs"]["windCaseV"]  = d["Fnumaxrs"]["windCaseII"] + self._Gamma3alphaDict["windCaseII"] * 2
        
        d["Fnumaxrs"]["ISMcaseVI"]  = d["Fnumaxrs"]["ISMcaseI"]   
        d["Fnumaxrs"]["ISMcaseVII"] = d["Fnumaxrs"]["ISMcaseII"]  
        d["Fnumaxrs"]["windCaseVI"] = d["Fnumaxrs"]["windCaseI"]  
        d["Fnumaxrs"]["windCaseVII"]= d["Fnumaxrs"]["windCaseII"] 
        
        return d
    
    def silence_vectorized_warnings(func):
        """Decorator to muzzle NumPy ufunc vectorized warnings at runtime. - Gemini"""
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings(), np.errstate(over='ignore', divide='ignore', invalid='ignore'):
                warnings.simplefilter("ignore", RuntimeWarning)
                return func(*args, **kwargs)
        return wrapper
    
    @silence_vectorized_warnings
    @np.vectorize
    def _tnub1eqnub2(tcross, nub1_tcross, nub2_tcross, alpha1, alpha2, postcross = True):
        """calculates the time at which two frequencies cross before or after 
        the crossing time when the power laws do not change from crossing time 
        to equality.
        """
        precross = not(postcross)
        
        if abs(alpha1 - alpha2) < 0.01: #if alpha1 == alpha2:
            return largeTime #np.nan
        else:
            with np.errstate(over = 'raise', divide = "raise", invalid = "raise"):
                try:
                    t = (nub2_tcross/nub1_tcross)**(1/(alpha1 - alpha2)) * tcross
                except (OverflowError, ZeroDivisionError, FloatingPointError, RuntimeWarning):
                    t = largeTime #np.nan #t = np.inf
                        
            if postcross and t > tcross:
                return t
            elif precross and t < tcross:
                return t
            else:
                return largeTime #np.nan
            
    @silence_vectorized_warnings
    @np.vectorize
    def _tnub1eqnub2double(tcross, nub1_tcross, nub2_tcross, alpha1a, tAtoB, alpha1b, alpha2, postcross = True):
        """calculates the time at which two frequencies cross before or after 
        the crossing time when the power laws changes once from crossing time 
        to equality for the first break frequency."""
        precross = not(postcross)
        
        if postcross and tAtoB < tcross:
            raise Exception("change in powerlaw should be after crossing time for postcross = True")
        elif precross and tAtoB > tcross and tAtoB < largeTime:
            raise Exception("change in powerlaw should be before crossing time for postcross = False")
        
        if abs(alpha1b - alpha2) < 0.01 or np.isnan(tAtoB) or tAtoB == largeTime: #if alpha1b == alpha2 or np.isnan(tAtoB):
            return largeTime #np.nan
        else:
            with np.errstate(over ='raise', divide = "raise", invalid = "raise"):
                try:
                    t = (nub2_tcross/nub1_tcross)**(1/(alpha1b - alpha2)) * tAtoB**((alpha1b - alpha1a)/(alpha1b - alpha2)) * tcross**((alpha1a - alpha2)/(alpha1b - alpha2))
                except (OverflowError, ZeroDivisionError, FloatingPointError, RuntimeWarning):
                    t = largeTime #t = np.nan  #t = np.inf
            
            if postcross and t > tAtoB:
                return t
            elif precross and t < tAtoB:
                return t
            else:
                return largeTime #np.nan