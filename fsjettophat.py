# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:14:15 2025

@author: rohdo
"""
import numpy as np
from spectrum import Spectrum
from cosmoCalc import lumDistLCDM
import warnings

class FSjetTopHat:
    """Granot and Sari 2002 table 2"""
    
    def __init__(self, nu, epse, epsB, n0, Astar, E52, tdays, z, p = 2.5, xie = 1, k = 0, tjet = np.inf, sjet = None):
        """"""
        self._nu = nu
        self._epse = epse * 1/xie #* xie
        self._epsB = epsB * 1/xie #* xie
        self._n0 = n0 * xie #* 1/xie
        self._Astar = Astar * xie #* 1/xie
        self._E52 = E52 * xie #* 1/xie
        self._tdays = tdays
        self._z = z
        self._dL28 = lumDistLCDM(z)*100/1e28
        self._p = p
        self._xie = xie
        self._k = k
        self._tjet = tjet
        self._sjet = sjet
        
        self._epseBar = self._epse * (self._p - 2)/(self._p - 1)
        
        self._fluxDict = self._buildFluxDict()
        self._breakFreqDict = self._buildBreakFreqDict()
        self._specDict = self._buildSpecDict()
        
        self._nuac, self._nusa, self._num, self._nuc = self._breakFreq()
        
    def spectrum(self):
        """"""
        k = self._k
        spec = Spectrum(self._nu, 1, self._nuac, self._nusa, self._num, self._nuc, Fnu1 = self._fluxDict[1][k], Fnu4 = self._fluxDict[4][k], Fnu7 = self._fluxDict[7][k])
        
        nu, Fnu = spec.spectrum()
        
        return nu, Fnu
        
    def _breakFreq(self):
        """"""
        
        d = self._breakFreqDict
        k = self._k
        #                                            nuac,     nusa,     num,      nuc.
        breakFreq = self._specDict[1] * np.array([      0,  d[1][k], d[2][k],  d[3][k]]) +\
                    self._specDict[2] * np.array([      0,  d[5][k], d[4][k],  d[3][k]]) +\
                    self._specDict[3] * np.array([      0,  d[6][k], d[4][k],  d[3][k]]) +\
                    self._specDict[4] * np.array([d[7][k],  d[8][k], d[9][k],        0]) +\
                    self._specDict[5] * np.array([d[7][k], d[10][k], d[9][k], d[11][k]])
        
        return breakFreq
    
    def _buildSpecDict(self):
        """"""
        d = self._breakFreqDict
        k = self._k

        D = { # not mutually exclusive
            1 : (d[1][k]  <  d[2][k] < d[3][k])            or (d[1][k] < d[9][k] < d[11][k]),
            2 : (d[4][k]  <  d[5][k] < d[3][k])            or (d[2][k] < d[1][k] < d[3][k]),
            3 : (d[4][k]  < d[6][k] and d[3][k] < d[6][k]) or (d[9][k]  < d[8][k] and d[3][k] < d[6][k]),
            4 :  d[11][k] <  d[8][k] < d[9][k],
            5 :  d[10][k] < d[11][k] < d[9][k],
        }
        
        truthArray = np.array(list(D.values()))
        
        if len(truthArray[truthArray == True]) != 1:
            warnings.warn("spectra identified were none or multiple \n {}".format(D), RuntimeWarning)
        
            if D[1] and D[2]: # 1 --> 2 transition handling
                warnings.warn("handling as 1 --> 2", RuntimeWarning)
                D[1] = False
            elif D[5] and D[1]: # 5 --> 1 transition handling
                warnings.warn("handling as 5 -- 1", RuntimeWarning)
                D[5] = False
            elif D[4] and D[3]: # 4 --> 3 transition handling
                warnings.warn("handling as 4 --> 3", RuntimeWarning)
                D[4] = False
            
        return D
    
    def _powers(self, zPlusOnePow, epseBarPow, epsBpow, n0pow, AstarPow, E52pow, tdaysPow, dL28pow):
        """"""
        return (1 + self._z)**zPlusOnePow * self._epseBar**epseBarPow * self._epsB**epsBpow *\
            self._n0**n0pow * self._Astar**AstarPow * self._E52**E52pow * self._tdays**tdaysPow * self._dL28**dL28pow
        
    def _buildFluxDict(self):
        """"""
        d = {
            1 : {
                0 : 0.647 * (self._p - 1)**(6/5)/((3 * self._p - 1) * (3 * self._p + 2)**(1/5)) * self._powers(1/2, -1, 2/5, 7/10,   0, 9/10,  1/2, -2),
                2 : 9.19  * (self._p - 1)**(6/5)/((3 * self._p - 1) * (3 * self._p + 2)**(1/5)) * self._powers(6/5, -1, 2/5,    0, 7/5,  1/5, -1/5, -2),
                },
            4 : {
                0 : 3.72  * (self._p - 1.79) * 1e15 * self._powers(7/2, 5, 1, -1/2,  0, 3/2, -5/2, -2),
                2 : 3.04  * (self._p - 1.79) * 1e15 * self._powers(  3, 5, 1,    0, -1,   2,   -2, -2),
                },
            7 : {
                0 : 5.27  * (3 * self._p - 1)**(11/15)/(3 * self._p + 2)**(11/15) * 1e3 * self._powers(-1/10, -11/5, -4/5, 1/10,   0, 3/10, 11/10, -2),
                2 : 3.76  * (3 * self._p - 1)**(11/15)/(3 * self._p + 2)**(11/15) * 1e3 * self._powers(    0, -11/5, -4/5,    0, 1/5,  1/5,     1, -2),
                },
        }
        
        return d
        
    def _buildBreakFreqDict(self):
        """"""
        d = {
            1 : {
                0 : 1.24 * (self._p - 1)**(3/5)/(3 * self._p + 2)**(3/5) * 1e9 * self._powers(  -1, -1, 1/5, 3/5,   0,  1/5,    0, 0),
                2 : 8.31 * (self._p - 1)**(3/5)/(3 * self._p + 2)**(3/5) * 1e9 * self._powers(-2/5, -1, 1/5,   0, 6/5, -2/5, -3/5, 0),
                },
            2 : {
                0 : 3.73 * (self._p - 0.67) * 1e15 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                2 : 4.02 * (self._p - 0.69) * 1e15 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                },
            3 : {
                0 : 6.37 * (self._p - 0.46) * 1e13 * np.exp(-1.16 * self._p) * self._powers(-1/2, 0, -3/2, -1,  0, -1/2, -1/2, 0),
                2 : 4.40 * (3.45 - self._p) * 1e10 * np.exp( 0.45 * self._p) * self._powers(-3/2, 0, -3/2,  0, -2,  1/2,  1/2, 0),
                },
            4 : {
                0 : 5.04 * (self._p - 1.22) * 1e16 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                2 : 8.08 * (self._p - 1.22) * 1e16 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                },
            5 : {
                0 : 3.59 * (4.03 - self._p) * 1e9  * np.exp(2.34 * self._p) * (self._powers(-(6 - self._p), 4 * (self._p - 1), self._p + 2, 4, 0,    self._p + 2, -(3 * self._p + 2), 0))**(1/(2 * (self._p + 4))),
                2 : 1.58 * (4.10 - self._p) * 1e10 * np.exp(2.16 * self._p) * (self._powers(-(2 - self._p), 4 * (self._p - 1), self._p + 2, 0, 8, -(2 - self._p), -3 * (self._p + 2), 0))**(1/(2 * (self._p + 4))),
                },
            6 : {
                0 : 3.23 * (self._p - 1.76) * 1e12 * (self._powers(-(7 - self._p), 4 * (self._p - 1), self._p - 1, 2, 0, self._p + 1, -3 * (self._p + 1), 0))**(1/(2 * (self._p + 5))),
                2 : 4.51 * (self._p - 1.73) * 1e12 * (self._powers(-(5 - self._p), 4 * (self._p - 1), self._p - 1, 0, 4, self._p - 1, -(3 * self._p + 5), 0))**(1/(2 * (self._p + 5))),
                },
            7 : {
                0 : 1.12 * (3 * self._p - 1)**(8/5)/(3 * self._p + 2)**(8/5) * 1e8 * self._powers(-13/10, -8/5, -2/5, 3/10,   0, -1/10, 3/10, 0),
                2 : 1.68 * (3 * self._p - 1)**(8/5)/(3 * self._p + 2)**(8/5) * 1e8 * self._powers(    -1, -8/5, -2/5,    0, 3/5,   2/5,    0, 0),
                },
            8 : {
                0 : 1.98e11 * self._powers(-1/2, 0, 0, 1/6,   0, 1/6, -1/2, 0),
                2 : 3.15e11 * self._powers(-1/3, 0, 0,   0, 1/3,   0, -2/3, 0),
                },
            9 : {
                0 : 3.94 * (self._p - 0.74) * 1e15 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                2 : 3.52 * (self._p - 0.31) * 1e15 * self._powers(1/2, 2, 1/2, 0, 0, 1/2, -3/2, 0),
                },
            10: {
                0 : 1.32e10 * self._powers(-1/2, 0, 6/5, 11/10,    0, 7/10, -1/2, 0),
                2 : 1.32e10 * self._powers(-1/2, 0, 6/5,     0, 11/5, 7/10, -1/2, 0), # TODO The breaks b 10; 11 involve PLS E see GS02 table 2
                },
            11: {
                0 : 5.86e12 * self._powers(-1/2, 0, -3/2, -1,  0, -1/2, -1/2, 0),
                2 : 5.86e12 * self._powers(-1/2, 0, -3/2,  0, -2, -1/2, -1/2, 0), # TODO The breaks b 10; 11 involve PLS E see GS02 table 2
                }
        }
        
        return d
    
    def _s_default(self, default):
        """"""
        if self._sjet is None:
            return default
        else:
            return self._sjet
    
    def _buildJetBreakAlphaDict(self):
        """"""
        s_default = 2
        
        d = {
            "F1"   : {
                     "alpha" : -2/5,
                     "s"     : self._s_default(s_default),
                     },
            "F4"   : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "F7"   : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu1"  : {
                     "alpha" : -1/5,
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu2"  : {
                     "alpha" : -2,
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu3"  : {
                     "alpha" : 0,
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu4"  : {
                     "alpha" : -2,
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu5"  : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu6"  : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu7"  : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu8"  : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu9"  : {
                     "alpha" : -2,
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu10" : {
                     "alpha" : np.nan, # TODO
                     "s"     : self._s_default(s_default), # TODO
                     },
            "nu11" : {
                     "alpha" : 0,
                     "s"     : self._s_default(s_default), # TODO
                     },
        }