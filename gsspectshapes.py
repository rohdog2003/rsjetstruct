# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:47:35 2025

@author: rohdo
"""

import numpy as np

class Spectrum:
    """Generates a spectrum given break frequency and peak according to the
    prescription in Granot and Sari 2002 (GS02).
    """
    
    def __init__(self, nu, Fnumax, nuac, nusa, num, nuc, p = 2.5, k = 0, Fnu1 = None, Fnu4 = None, Fnu7 = None, cutoff = False):
        """
        NOTE: Fnumax is the extrapolated peak
        NOTE: only nu can be a vector and other inputs must be scalars
        """
        self._nu = nu
        self._Fnumax = Fnumax
        self._nuac = nuac
        self._nusa = nusa
        self._num = num
        self._nuc = nuc
        self._p = p
        self._k = k
        self._Fnu1 = Fnu1
        self._Fnu4 = Fnu4
        self._Fnu7 = Fnu7
        self._cutoff = cutoff
        
        self._errDict = Spectrum._buildErrDict()
    
    def spectrum(self): # FIXME something bad will probably happen if any of the frequencies are equal
        """"""
        if self._nuac <= self._nusa <= self._num <= self._nuc: # spectrum 1
            if self._Fnu1 is None:
                # Fnu1 = self._Fnumax/self._spectrum1(self._num, 1); # TODO
                Fnu1 = self._Fnumax * (self._nusa/self._num)**self._getSlope(2)[0]
            else:
                Fnu1 = self._Fnu1
            
            spec = self._spectrum1(self._nu, Fnu1)
            
        elif self._nuac <= self._num <= self._nusa <= self._nuc: # spectrum 2
            if self._Fnu4 is None:
                #Fnu4 = self._Fnumax/self._spectrum2(self._nusa, 1)
                Fnu4 = self._Fnumax * (self._num/self._nusa)**self._getSlope(5)[0]
            else:
                Fnu4 = self._Fnu4
            
            spec = self._spectrum2(self._nu, Fnu4)
            
        elif self._nuac <= self._nusa and self._num <= self._nusa and self._nuc <= self._nusa: # spectrum 3
            if self._Fnu4 is None:
                #Fnu4 = self._Fnumax/self._spectrum3(self._nusa, 1)
                Fnu4 = self._Fnumax * (self._num/self._nusa)**self._getSlope(5)[0]
            else:
                Fnu4  = self._Fnu4
            
            spec = self._spectrum3(self._nu, Fnu4)
            
        elif self._nuac <= self._nusa and self._nuc <= self._nusa and self._nusa <= self._num: # spectrum 4
            if self._Fnu7 is None:
                #Fnu7 = self._Fnumax/self._spectrum4(self._nusa, 1)
                Fnu7 = self._Fnumax * (self._nuac/self._nusa)**self._getSlope(8)[0]
            else:
                Fnu7 = self._Fnu7
            
            spec = self._spectrum4(self._nu, Fnu7)
            
        elif self._nuac <= self._nusa <= self._nuc <= self._num: # spectrum 5
            if self._Fnu7 is None:
                #Fnu7 = self._Fnumax/self._spectrum5(self._nuc, 1)
                Fnu7 = self._Fnumax * (self._nusa/self._nuc)**self._getSlope(11)[0] * (self._nuac/self._nusa)**self._getSlope(10)[0]
            else:
                Fnu7 = self._Fnu7
            
            spec = self._spectrum5(self._nu, Fnu7)
        
        else:
            raise Exception("nuac must be smaller than nusa")
        
        return self._nu, spec
    
    def _Fnub(nu, nub, Fnub, beta1, beta2, s):
        """flux density near a spectral break GS02 (1).
        """
        return Fnub * ((nu/nub)**(-s * beta1) + (nu/nub)**(-s * beta2))**(-1/s)\
               #* 0.5**(-1/s) # TODO correction (?) lowered chi2 of fit
    
    def _Fnu4(nu, nu4, Fnu4, beta1, beta2, s):
        """flux density near a spectral break for frequency 4 GS02 (3).
        """
        phi4 = nu/nu4
        
        return Fnu4 * (phi4**2 * np.exp(-s * phi4**(2/3)) + phi4**(5/2))\
               #/(np.exp(-s) + 1) # TODO correction (?) lowered chi2 of fit
    
    def _tildeFnub(nu, nub, beta1, beta2, s):
        """GS02 (4).
        """
        return (1 + (nu/nub)**(s * (beta1 - beta2)))**(-1/s)
    
    def _spectrum1(self, nu, Fnu1):
        """GS02 (5).
        """
        
        return Spectrum._Fnub(nu, self._nub(1), Fnu1, *self._getSlope(1)) *\
               Spectrum._tildeFnub(nu, self._nub(2), *self._getSlope(2)) *\
               (np.logical_not(self._cutoff) * Spectrum._tildeFnub(nu, self._nub(3), *self._getSlope(3)) + self._cutoff * np.exp(-nu/self._nub(3)))
        
    def _spectrum2(self, nu, Fnu4):
        """GS02 (6).
        """
        return Spectrum._Fnu4(nu, self._nub(4), Fnu4, *self._getSlope(4)) *\
               Spectrum._tildeFnub(nu, self._nub(5), *self._getSlope(5)) *\
               (np.logical_not(self._cutoff) * Spectrum._tildeFnub(nu, self._nub(3), *self._getSlope(3)) + self._cutoff * np.exp(-nu/self._nub(3)))
        
    def _spectrum3(self, nu, Fnu4):
        """GS02 (7).
        """
        return Spectrum._Fnu4(nu, self._nub(4), Fnu4, *self._getSlope(4)) *\
               (np.logical_not(self._cutoff) * Spectrum._tildeFnub(nu, self._nub(6), *self._getSlope(6)) + self._cutoff * np.exp(-nu/self._nub(3)))
        
    def _spectrum4(self, nu, Fnu7):
        """GS02 (8).
        """
        return Spectrum._Fnub(nu, self._nub(7), Fnu7, *self._getSlope(7)) *\
               (np.logical_not(self._cutoff) * Spectrum._tildeFnub(nu, self._nub(8), *self._getSlope(8)) *\
               Spectrum._tildeFnub(nu, self._nub(9), *self._getSlope(9)) +\
               self._cutoff * np.exp(-nu/self._nub(11)))
        
    def _spectrum5(self, nu, Fnu7):
        """GS02 (9).
        """
        return Spectrum._Fnub(nu, self._nub(7), Fnu7, *self._getSlope(7)) *\
               Spectrum._tildeFnub(nu, self._nub(10), *self._getSlope(10)) *\
               (np.logical_not(self._cutoff) * Spectrum._tildeFnub(nu, self._nub(11), *self._getSlope(11)) *\
               Spectrum._tildeFnub(nu, self._nub(9), *self._getSlope(9)) + self._cutoff * np.exp(-nu/self._nub(11)))
    
    def _nub(self, b):
        """"""
        if b==1:
            return self._nusa
        elif b==2:
            return self._num
        elif b==3:
            return self._nuc
        elif b==4:
            return self._num
        elif b==5:
            return self._nusa
        elif b==6:
            return self._nusa
        elif b==7:
            return self._nuac
        elif b==8:
            return self._nusa
        elif b==9:
            return self._num
        elif b==10:
            return self._nusa
        elif b==11:
            return self._nuc
        else:
            raise Exception("b must be an integer between 1 (inclusive) and 11 (inclusive)")
    
    def _s(self, s1, s2):
        """returns the first input for ISM and second for wind.
        """
        if self._k == 0:
            return s1
        elif self._k == 2:
            return s2
    
    def _getSlope(self, b):
        """returns the spectral indices and smoothing factor according to GS02 
        table 2.
        """
        if b==1:
            beta1 = 2;               beta2 = 1/3
            
            s = self._s(1.64,\
                        1.06)
        elif b==2:
            beta1 = 1/3;             beta2 = (1 - self._p)/2
            
            s = self._s(1.84 - 0.4 * self._p,\
                        1.76 - 0.38 * self._p)
        elif b==3:
            beta1 = (1 - self._p)/2; beta2 = -self._p/2
            
            s = self._s(1.15 - 0.06 * self._p,\
                        0.80 - 0.03 * self._p)
        elif b==4:
            beta1 = 2;               beta2 = 5/2
            
            s = self._s(3.44 * self._p - 1.41,\
                        3.63 * self._p - 1.60)
        elif b==5:
            beta1 = 5/2;             beta2 = (1 - self._p)/2
            
            s = self._s(1.47 - 0.21 * self._p,\
                        1.25 - 0.18 * self._p)
        elif b==6:
            beta1 = 5/2;             beta2 = -self._p/2
            
            s = self._s(0.94 - 0.14 * self._p,\
                        1.04 - 0.16 * self._p)
        elif b==7:
            beta1 = 2;               beta2 = 11/8
            
            s = self._s(1.99 - 0.04 * self._p,\
                        1.97 - 0.04 * self._p)
        elif b==8:
            beta1 = 11/8;            beta2 = -1/2
            
            s = self._s(0.907,\
                        0.893)
        elif b==9:
            beta1 = -1/2;            beta2 = -self._p/2
            
            s = self._s(3.34 - 0.82 * self._p,\
                        3.68 - 0.89 * self._p)
        elif b==10:
            beta1 = 11/8;            beta2 = 1/3
            
            s = self._s(1.213,\
                        1.213) # table 2 of GS02 excludes so taken to be same as ISM
        elif b==11:
            beta1 = 1/3;             beta2 = -1/2
            
            s = self._s(0.597,\
                        0.597) # table 2 of GS02 excludes so taken to be same as ISM
        else:
            raise Exception("b must be an integer between 1 (inclusive) and 11 (inclusive)")
            
        return beta1, beta2, s
    
    def _buildErrDict():
        """Gets the systematic error from each power law break.
        """
        #b   k
        errDict =\
        {
        1  : {
             0 : 0.0668,
             2 : 0.0102
             },
        2  : {
             0 : 0.059,
             2 : 0.072
             },
        3  : {
             0 : 0.019,
             2 : 0.044
             },
        4  : {
             0 : 0.007,
             2 : 0.018
             },
        5  : {
             0 : 0.059,
             2 : 0.072
             },
        6  : {
             0 : 0.124,
             2 : 0.110
             },
        7  : {
             0 : 0.019,
             2 : 0.019
             },
        8  : {
             0 : 0.0171,
             2 : 0.0229
             },
        9  : {
             0 : 0.045,
             2 : 0.042
             },
        10 : {
             0 : 0.0522,
             2 : 1.000 # table 2 of GS02 excludes so error is taken to be large
             },
        11 : {
             0 : 0.0055,
             2 : 1.000 # table 2 of GS02 excludes so error is table to be large
             }
        }
        
        return errDict
    
    def _getError(self, b):
        """"""
        return self._errDict[b][self._k]
