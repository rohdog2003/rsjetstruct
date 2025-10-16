# -*- coding: utf-8 -*-
"""
Created on Wed May 28 10:19:34 2025

@author: rohdo
"""
import numpy as np
from .gsspectshapes import Spectrum
#from obsfluxmax import *
import warnings

smallNum = 1e-50 # a very small number close to zero
def obsFluxMax(Fnumax_nossa, nuac, nusa, num, nuc, p):
    """computes the observed maximum flux from the theoretical maximum if no
    synchrotron self absorption were to occur.
    """
    F1 = Fnumax_nossa * (nuac <= nusa <= num <= nuc) # spectrum 1
    F2 = Fnumax_nossa * (nusa/num)**(-(p - 1)/2)*(nuac <= num <= nusa <= nuc)     # spectrum 2
    F3 = Fnumax_nossa * (nuc/num)**(-(p - 1)/2) * (nusa/nuc)**(-p/2) * (nuac <= nusa and num <= nusa and nuc <= nusa and num <= nuc) + \
         Fnumax_nossa * (num/nuc)**(-1/2) * (nusa/num)**(-p/2) * (nuac <= nusa and num <= nusa and nuc <= nusa and num > nuc)  # spectrum 3
    F4 = Fnumax_nossa * (nusa/nuc)**(-1/2) * (nuac <= nusa and nuc <= nusa and nusa <= num) # spectrum 4
    F5 = Fnumax_nossa * (nuac <= nusa <= nuc <= num) # spectrum 5
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
    
    def __init__(self, tobs, nu, tcross, Fnumaxrs_tcross, numrs_tcross, nucutrs_tcross, nuars_tcross, keps, kGamma,\
                 k, p, g = None, tjet = np.inf):
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
        
        self._tfrac = self._tobs/self._tcross
        self._a = self._compute_a()
        self._A = self._compute_A()
        
        self._Gamma3alphaDict = self._buildGamma3alphaDict()
        self._alphaDict = self._buildAlphaDict()
        # post crossing time equalities for nuars and numrs slow cooling 
        # TODO add case c
        self._tnuarseqnumrsPostCrossISMcaseIa     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIa"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
        self._tnuarseqnumrsPostCrossISMcaseIb     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
        self._tnuarseqnumrsPostCrossISMcaseIIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIIa"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
        self._tnuarseqnumrsPostCrossISMcaseIIb    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIa"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIb    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIIa   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIIa"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,     self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
        # post crossing time equalities for nuars and nucutrs slow cooling
        self._tnuarseqnucutrsPostCrossISMcaseIa   = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIa"],\
                                                                                                 self._tnuarseqnumrsPostCrossISMcaseIa,      self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
        self._tnuarseqnucutrsPostCrossISMcaseIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
        self._tnuarseqnucutrsPostCrossISMcaseIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIc"],    self._alphaDict["nucutrs"]["ISMcaseI"],    postcross = True)
        self._tnuarseqnucutrsPostCrossISMcaseIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIa"],\
                                                                                                 self._tnuarseqnumrsPostCrossISMcaseIIa,     self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
        self._tnuarseqnucutrsPostCrossISMcaseIIb  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
        self._tnuarseqnucutrsPostCrossISMcaseIIc  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,   self._alphaDict["nuars"]["ISMcaseIIc"],   self._alphaDict["nucutrs"]["ISMcaseII"],   postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIa   = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIa"],\
                                                                                                  self._tnuarseqnumrsPostCrossWindCaseIa,    self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIb   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIc"],   self._alphaDict["nucutrs"]["windCaseI"],   postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIa"],\
                                                                                                  self._tnuarseqnumrsPostCrossWindCaseIIa,   self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIIb  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
        self._tnuarseqnucutrsPostCrossWindCaseIIc  =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIc"],  self._alphaDict["nucutrs"]["windCaseII"],  postcross = True)
        # post crossing time equalities for nuars and numrs slow cooling (double crossing)
        self._tnuarseqnumrsPostCrossISMcaseIc      = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIc"],\
                                                                                                  self._tnuarseqnucutrsPostCrossISMcaseIc,   self._alphaDict["nuars"]["ISMcaseIb"],    self._alphaDict["numrs"]["ISMcaseI"],      postcross = True)
        self._tnuarseqnumrsPostCrossISMcaseIIc     = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIIc"],\
                                                                                                  self._tnuarseqnucutrsPostCrossISMcaseIIc,  self._alphaDict["nuars"]["ISMcaseIIb"],   self._alphaDict["numrs"]["ISMcaseII"],     postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIc     = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIc"],\
                                                                                                  self._tnuarseqnucutrsPostCrossWindCaseIc,  self._alphaDict["nuars"]["windCaseIb"],   self._alphaDict["numrs"]["windCaseI"],     postcross = True)
        self._tnuarseqnumrsPostCrossWindCaseIIc    = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIIc"],\
                                                                                                  self._tnuarseqnucutrsPostCrossWindCaseIIc, self._alphaDict["nuars"]["windCaseIIb"],  self._alphaDict["numrs"]["windCaseII"],    postcross = True)
        # pre crossing time equalities for nuars and numrs slow cooling
        self._tnuarseqnumrsPreCrossISMcaseIIIa     =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["ISMcaseIIIa"],  self._alphaDict["numrs"]["ISMcaseIII"],    postcross = False)
        self._tnuarseqnumrsPreCrossISMcaseIIIb     =       np.nan # always np.nan
        self._tnuarseqnumrsPreCrossISMcaseIIIc     =       np.nan # always np.nan
        self._tnuarseqnumrsPreCrossWindCaseIIIa    =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._numrs_tcross,    self._alphaDict["nuars"]["windCaseIIIa"], self._alphaDict["numrs"]["windCaseIII"],   postcross = False)
        self._tnuarseqnumrsPreCrossWindCaseIIIb    =       np.nan # always np.nan
        self._tnuarseqnumrsPreCrossWindCaseIIIc    =       np.nan # always np.nan
        # pre crossing time equalities for nuars and nucutrs slow cooling
        self._tnuarseqnucutrsPreCrossISMcaseIIIa   =       np.nan # always np.nan
        self._tnuarseqnucutrsPreCrossISMcaseIIIb   =       np.nan # always np.nan
        self._tnuarseqnucutrsPreCrossISMcaseIIIc   =       RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["ISMcaseIIIc"],  self._alphaDict["nucutrs"]["ISMcaseIII"],  postcross = False)
        self._tnuarseqnucutrsPreCrossWindCaseIIIa  = RSjetStruct._tnub1eqnub2double(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIIa"],\
                                                                                                  self._tnuarseqnumrsPreCrossWindCaseIIIa,   self._alphaDict["nuars"]["windCaseIIIb"], self._alphaDict["nucutrs"]["windCaseIII"], postcross = False) # FIXME sometimes np.nan when it shouldn't be
        self._tnuarseqnucutrsPreCrossWindCaseIIIb =        RSjetStruct._tnub1eqnub2(self._tcross, self._nuars_tcross, self._nucutrs_tcross,  self._alphaDict["nuars"]["windCaseIIIb"], self._alphaDict["nucutrs"]["windCaseIII"], postcross = False)
        self._tnuarseqnucutrsPreCrossWindCaseIIIc =        np.nan # always np.nan
        
        self._Fnumaxrs = self.Fnumaxrs()
        self._numrs = self.numrs()
        self._nucutrs = self.nucutrs()
        self._nuars = self.nuars()
    
    def spectrum(self, diagnostic=False):
        """TODO
        
        Returns
        -------
        nu : float
            TODO
        
        Fnu : float
            TODO
        """
        return RSjetStruct._spectrum(self._tobs, self._nu, self._tcross, self._Fnumaxrs, self._numrs, self._nucutrs, self._nuars, self._p, self._k, diagnostic=diagnostic)
    
    @np.vectorize # TODO figure out a way to do without np.vectorize here to take advantage of spectrum.py's vectorization
    def _spectrum(_tobs, _nu, _tcross, _Fnumaxrs, _numrs, _nucutrs, _nuars, _p, _k, diagnostic = False):
        """"""
        _Fnutruemaxrs = obsFluxMax(_Fnumaxrs, smallNum, _nuars, _numrs, _nucutrs, p = _p)
        
        spec = Spectrum(_nu, _Fnutruemaxrs, smallNum, _nuars, _numrs, _nucutrs, p = _p, k = _k)
        
        nu, Fnu = spec.spectrum()

        if (diagnostic):
            return _nuars, _numrs, _nucutrs, _Fnutruemaxrs
        else: 
            return nu, np.where(nu < _nucutrs, Fnu,\
                                               Fnu * np.exp(-_nu/_nucutrs)**(_tobs > _tcross))
        
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
        nuars_ISMcaseIa   = self._compute_caseA(observable, ISMscale_caseIa,   self._tnuarseqnumrsPostCrossISMcaseIa,   self._tnuarseqnucutrsPostCrossISMcaseIa,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIa,  self._tnuarseqnucutrsPreCrossISMcaseIIIa,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        nuars_ISMcaseIb   = self._compute_caseB(observable, ISMscale_caseIb,   self._tnuarseqnumrsPostCrossISMcaseIb,   self._tnuarseqnucutrsPostCrossISMcaseIb,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIb,  self._tnuarseqnucutrsPreCrossISMcaseIIIb,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        nuars_ISMcaseIc   = self._compute_caseC(observable, ISMscale_caseIc,   self._tnuarseqnumrsPostCrossISMcaseIc,   self._tnuarseqnucutrsPostCrossISMcaseIc,   "ISMcaseIa",   "ISMcaseIb",   "ISMcaseIc",   self._tnuarseqnumrsPreCrossISMcaseIIIc,  self._tnuarseqnucutrsPreCrossISMcaseIIIc,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        nuars_ISMcaseIIa  = self._compute_caseA(observable, ISMscale_caseIIa,  self._tnuarseqnumrsPostCrossISMcaseIIa,  self._tnuarseqnucutrsPostCrossISMcaseIIa,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIa,  self._tnuarseqnucutrsPreCrossISMcaseIIIa,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        nuars_ISMcaseIIb  = self._compute_caseB(observable, ISMscale_caseIIb,  self._tnuarseqnumrsPostCrossISMcaseIIb,  self._tnuarseqnucutrsPostCrossISMcaseIIb,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIb,  self._tnuarseqnucutrsPreCrossISMcaseIIIb,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        nuars_ISMcaseIIc  = self._compute_caseC(observable, ISMscale_caseIIc,  self._tnuarseqnumrsPostCrossISMcaseIIc,  self._tnuarseqnucutrsPostCrossISMcaseIIc,  "ISMcaseIIa",  "ISMcaseIIb",  "ISMcaseIIc",  self._tnuarseqnumrsPreCrossISMcaseIIIc,  self._tnuarseqnucutrsPreCrossISMcaseIIIc,  "ISMcaseIIIa",  "ISMcaseIIIb",  "ISMcaseIIIc")
        
        nuars_windCaseIa  = self._compute_caseA(observable, windScale_caseIa,  self._tnuarseqnumrsPostCrossWindCaseIa,  self._tnuarseqnucutrsPostCrossWindCaseIa,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIa, self._tnuarseqnucutrsPreCrossWindCaseIIIa, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        nuars_windCaseIb  = self._compute_caseB(observable, windScale_caseIb,  self._tnuarseqnumrsPostCrossWindCaseIb,  self._tnuarseqnucutrsPostCrossWindCaseIb,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIb, self._tnuarseqnucutrsPreCrossWindCaseIIIb, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        nuars_windCaseIc  = self._compute_caseC(observable, windScale_caseIc,  self._tnuarseqnumrsPostCrossWindCaseIc,  self._tnuarseqnucutrsPostCrossWindCaseIc,  "windCaseIa",  "windCaseIb",  "windCaseIc",  self._tnuarseqnumrsPreCrossWindCaseIIIc, self._tnuarseqnucutrsPreCrossWindCaseIIIc, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        nuars_windCaseIIa = self._compute_caseA(observable, windScale_caseIIa, self._tnuarseqnumrsPostCrossWindCaseIIa, self._tnuarseqnucutrsPostCrossWindCaseIIa, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIa, self._tnuarseqnucutrsPreCrossWindCaseIIIa, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        nuars_windCaseIIb = self._compute_caseB(observable, windScale_caseIIb, self._tnuarseqnumrsPostCrossWindCaseIIb, self._tnuarseqnucutrsPostCrossWindCaseIIb, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIb, self._tnuarseqnucutrsPreCrossWindCaseIIIb, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        nuars_windCaseIIc = self._compute_caseC(observable, windScale_caseIIc, self._tnuarseqnumrsPostCrossWindCaseIIc, self._tnuarseqnucutrsPostCrossWindCaseIIc, "windCaseIIa", "windCaseIIb", "windCaseIIc", self._tnuarseqnumrsPreCrossWindCaseIIIc, self._tnuarseqnucutrsPreCrossWindCaseIIIc, "windCaseIIIa", "windCaseIIIb", "windCaseIIIc")
        
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
        return np.where(self._ISM, i1, i2)
    
    def _casePreOrPost(self, i1, i2):
        """"""
        return np.where(self._tobs < self._tcross, i1, i2)
        
    def _caseIorII(self, i1, i2):
        """"""
        return np.where(self._kGamma < 1, i1, i2)
    
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
        nuars_caseA = np.where(nuars_caseA < self._numrs, nuars_caseA,\
                                                          observable *\
                                                          (teqmcaseA/self._tcross)**self._alphaDict["nuars"][caseAstr] *\
                                                          (self._tobs/teqmcaseA)**self._alphaDict["nuars"][caseBstr])
        nuars_caseA = self._BCmerge(nuars_caseA, caseBstr, caseCstr, above = False)
        nuars_caseA = np.where((nuars_caseA <= self._nucutrs), nuars_caseA,\
                                                              observable *\
                                                              (teqmcaseA/self._tcross)**self._alphaDict["nuars"][caseAstr] *\
                                                              (teqcutcaseA/teqmcaseA)**self._alphaDict["nuars"][caseBstr] *\
                                                              (self._tobs/teqcutcaseA)**self._alphaDict["nuars"][caseCstr])
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
        nuars_caseB = np.where(nuars_caseB > self._numrs, nuars_caseB,\
                                                          observable *\
                                                          (teqmcaseB/self._tcross)**self._alphaDict["nuars"][caseBstr] *\
                                                          (self._tobs/teqmcaseB)**self._alphaDict["nuars"][caseAstr])
        nuars_caseB = self._BCmerge(nuars_caseB, caseBstr, caseCstr, above = False)
        nuars_caseB = np.where((nuars_caseB <= self._nucutrs), nuars_caseB,\
                                                              observable *\
                                                              (teqcutcaseB/self._tcross)**self._alphaDict["nuars"][caseBstr] *\
                                                              (self._tobs/teqcutcaseB)**self._alphaDict["nuars"][caseCstr])
    
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
        nuars_caseC = np.where(nuars_caseC >= self._nucutrs, nuars_caseC,\
                                                             observable *\
                                                             (teqcutcaseC/self._tcross)**self._alphaDict["nuars"][caseCstr] *\
                                                             (self._tobs/teqcutcaseC)**self._alphaDict["nuars"][caseBstr])
        nuars_caseC = np.where(nuars_caseC > self._numrs, nuars_caseC,\
                                                          observable *\
                                                          (teqcutcaseC/self._tcross)**self._alphaDict["nuars"][caseCstr] *\
                                                          (teqmcaseC/teqcutcaseC)**self._alphaDict["nuars"][caseBstr] *\
                                                          (self._tobs/teqmcaseC)**self._alphaDict["nuars"][caseCstr])
        
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
        return np.where(self._ISM, np.where(self._kGamma < 1, observable * ISMscale_caseIV,\
                                                              observable * ISMscale_caseV),\
                                   np.where(self._kGamma < 1, observable * windScale_caseIV,\
                                                              observable * windScale_caseV))
    
    def _caseJet(self, i1, i2):
        """"""
        return np.where(self._tobs < self._tjet, i1, i2)
    
    def Fnumaxrs(self): # TODO add jet break
        """"""
        ISMscale_caseI = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseI"])
        ISMscale_caseII = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseII"])
        ISMscale_caseIII = self._tfrac**(self._alphaDict["Fnumaxrs"]["ISMcaseIII"])
        
        windScale_caseI = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseI"])
        windScale_caseII = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseII"])
        windScale_caseIII = self._tfrac**(self._alphaDict["Fnumaxrs"]["windCaseIII"])
        
        with np.errstate(divide = "raise"):
            try:
                ISMscale_caseIV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseI"]) *\
                                  (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseIV"])
                ISMscale_caseV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["ISMcaseII"]) *\
                                 (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["ISMcaseV"])
                                 
                windScale_caseIV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseI"]) *\
                                   (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseIV"])
                windScale_caseV = (self._tjet/self._tcross)**(self._alphaDict["Fnumaxrs"]["windCaseII"]) *\
                                  (self._tobs/self._tjet)**(self._alphaDict["Fnumaxrs"]["windCaseV"])
            except FloatingPointError:
                ISMscale_caseIV = np.nan
                ISMscale_caseV = np.nan
                windScale_caseIV = np.nan
                windScale_caseV = np.nan
                
        
        Fnumaxrs_preJet = self._cases(self._Fnumaxrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        Fnumaxrs_postJet = self._compute_casePostJet(self._Fnumaxrs_tcross, ISMscale_caseIV, ISMscale_caseV, windScale_caseIV, windScale_caseV)
        
        return self._caseJet(Fnumaxrs_preJet, Fnumaxrs_postJet)
        
    def numrs(self):
        """"""
        ISMscale_caseI = self._tfrac**(self._alphaDict["numrs"]["ISMcaseI"])
        ISMscale_caseII = self._tfrac**(self._alphaDict["numrs"]["ISMcaseII"])
        ISMscale_caseIII = self._tfrac**(self._alphaDict["numrs"]["ISMcaseIII"])
        
        windScale_caseI = self._tfrac**(self._alphaDict["numrs"]["windCaseI"])
        windScale_caseII = self._tfrac**(self._alphaDict["numrs"]["windCaseII"])
        windScale_caseIII = self._tfrac**(self._alphaDict["numrs"]["windCaseIII"])
        
        return self._cases(self._numrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        
    def nucutrs(self):
        """"""
        ISMscale_caseI = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseI"])
        ISMscale_caseII = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseII"])
        ISMscale_caseIII = self._tfrac**(self._alphaDict["nucutrs"]["ISMcaseIII"])
        
        windScale_caseI = self._tfrac**(self._alphaDict["nucutrs"]["windCaseI"])
        windScale_caseII = self._tfrac**(self._alphaDict["nucutrs"]["windCaseII"])
        windScale_caseIII = self._tfrac**(self._alphaDict["nucutrs"]["windCaseIII"])
        
        return self._cases(self._nucutrs_tcross, ISMscale_caseI, ISMscale_caseII, ISMscale_caseIII, windScale_caseI, windScale_caseII, windScale_caseIII)
        
    def nuars(self):
        """"""
        ISMscale_caseIa = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIa"])
        ISMscale_caseIb = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIb"])
        ISMscale_caseIc = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIc"])
        ISMscale_caseIIa = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIa"])
        ISMscale_caseIIb = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIb"])
        ISMscale_caseIIc = self._tfrac**(self._alphaDict["nuars"]["ISMcaseIIc"])
        
        windScale_caseIa = self._tfrac**(self._alphaDict["nuars"]["windCaseIa"])
        windScale_caseIb = self._tfrac**(self._alphaDict["nuars"]["windCaseIb"])
        windScale_caseIc = self._tfrac**(self._alphaDict["nuars"]["windCaseIc"])
        windScale_caseIIa = self._tfrac**(self._alphaDict["nuars"]["windCaseIIa"])
        windScale_caseIIb = self._tfrac**(self._alphaDict["nuars"]["windCaseIIb"])
        windScale_caseIIc = self._tfrac**(self._alphaDict["nuars"]["windCaseIIc"])
        
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
            "windCaseIII" : np.nan
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
        """Case I is for k_Gamma < 1 and case II is for k_Gamma >= 1 both for 
        time between t_cross and t_jet. Case III is for time less than t_cross.
        Case IV is for k_Gamma < 1 and case V is for k_Gamma >= 1 both for 
        time greater than t_jet. Case a is nu_a < nu_m < nu_c, case b 
        is nu_m < nu_a < nu_c, case c is nu_m < nu_c < nu_a, case d is
        nu_a < nu_c < nu_m, case e is nu_c < n_a < nu_m, case f is 
        nu_c < n_m < n_a.
        
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
            
        return d
    
    @np.vectorize
    def _tnub1eqnub2(tcross, nub1_tcross, nub2_tcross, alpha1, alpha2, postcross = True):
        """calculates the time at which two frequencies cross before or after 
        the crossing time when the power laws do not change from crossing time 
        to equality.
        """
        precross = not(postcross)
        
        if alpha1 == alpha2:
            return np.nan
        else:
            try:
                t = (nub2_tcross/nub1_tcross)**(1/(alpha1 - alpha2)) * tcross
            except (OverflowError, ZeroDivisionError):
                t = np.inf
            
            if postcross and t > tcross:
                return t
            elif precross and t < tcross:
                return t
            else:
                return np.nan
    
    @np.vectorize
    def _tnub1eqnub2double(tcross, nub1_tcross, nub2_tcross, alpha1a, tAtoB, alpha1b, alpha2, postcross = True):
        """calculates the time at which two frequencies cross before or after 
        the crossing time when the power laws changes once from crossing time 
        to equality for the first break frequency."""
        precross = not(postcross)
        
        if postcross and tAtoB < tcross:
            raise Exception("change in powerlaw should be after crossing time for postcross = True")
        elif precross and tAtoB > tcross:
            raise Exception("change in powerlaw should be before crossing time for postcross = False")
        
        if alpha1b == alpha2 or np.isnan(tAtoB):
            return np.nan
        else:
            try:
                t = (nub2_tcross/nub1_tcross * tAtoB**(alpha1b - alpha1a) * tcross**(alpha1a - alpha2))**(1/(alpha1b - alpha2)) # FIXME finding incorrect crossing time for precross case a num=nucut
            except (OverflowError, ZeroDivisionError):
                t = np.inf
            
            if postcross and t > tAtoB:
                return t
            elif precross and t < tAtoB:
                return t
            else:
                return np.nan
            
            
        