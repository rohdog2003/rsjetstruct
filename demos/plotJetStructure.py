# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 10:50:07 2025

@author: rohdo
"""

import matplotlib.pyplot as plt
import numpy as np
from plotTools import c1, c2, c3, c4, c5, c6, c7, multicolored_text

thetaj = np.pi/180 * 0.6 # Zhang, Wang, Zheng 2024 section 2
THETA = np.pi/180 * 1.46 # Laskar et al. 2023 table 6
II = 0.005
k = -1

def jetstruct(theta, thetaj = thetaj, THETA = THETA, I = 1, II = II, k = k):
    """"""
    return np.full_like(theta, I) * ((0 < theta) & (theta < thetaj)) +\
           II * theta**(k) * ((thetaj < theta) & (theta < THETA))
    
def jetstruct_plcontinued(theta, thetaj = thetaj, THETA = THETA, I = 1, II = II, k = k):
    """"""
    return II * theta**(k) * ((0 < theta) & (theta < THETA))
    
def jetstruct_plcutoff(theta, thetaj = thetaj, THETA = THETA, I = 1, II = II, k = k):
    """"""
    return II * theta**(k) * ((thetaj < theta) & (theta < THETA))

def jettophat(theta, thetaj = thetaj, THETA = THETA, II = II, k = k):
    """"""
    return np.full_like(theta, II) * thetaj**(k) * ((0 < theta) & (theta < THETA))

theta = np.geomspace(0.007, THETA * 1.1, 10000)

#plt.rcParams["text.usetex"] = False

plt.xlabel(r"Angle From Jet Axis of Symmetry $\log(\theta)$")
plt.xscale("log")
plt.ylabel(r"Energy Per Solid Angle $\log(ϵ)$" + " \n or \n " + r"Initial Lorentz Factor $\log(\Gamma_0)$")
plt.yscale("log")
plt.yticks([])
plt.yticks([], minor = True)
plt.title("Two Component Structured Jet Model")
plt.text(1.45e-2, 0.35, r"$\leftarrow ϵ\propto\theta^{-k_ϵ}$ or $\Gamma_0\propto\theta^{-k_\Gamma}$")
#cd = {"default" : "k", "k_ϵ" : c1, "k_\Gamma" : c1}
#multicolored_text(1.45e-2, 0.35, r"$\leftarrow ϵ\propto\theta^{-k_ϵ}$ or $\Gamma_0\propto\theta^{-k_\Gamma}$", cd)
plt.axvline(thetaj, color = "k", linestyle = "-.", label = r"$\theta_j={:.3f}$".format(thetaj))
plt.axvline(THETA,  color = "k",  linestyle = ":", label = r"$\Theta={:.3f}$".format(THETA))
plt.plot(theta, jettophat(theta), color = c5, linewidth = 3, linestyle = "--", label = "top hat")
plt.plot(theta, jetstruct(theta), color = c6, linewidth = 3, label = "structured")
plt.legend()

plt.savefig("plots/jetStructure", dpi = 1000)
plt.savefig("plots/jetStructure.svg")

for artist in plt.gca().lines:
    artist.remove()
    
plt.title("Structured Jet Model")
#plt.axvline(thetaj, color = "k", linestyle = "-.", label = r"$\theta_j={:.3f}$".format(thetaj))
plt.axvline(THETA,  color = "k",  linestyle = ":", label = r"$\Theta={:.3f}$".format(THETA))
plt.plot(theta, jettophat(theta), color = c5, linewidth = 3, linestyle = "--", label = "top hat")
plt.plot(theta, jetstruct_plcontinued(theta), color = c6, linewidth = 3, label = "structured")
plt.legend()

plt.savefig("plots/jetStructure_plcontinued", dpi = 1000)
plt.savefig("plots/jetStructure_plcontinued.svg")

for artist in plt.gca().lines:
    artist.remove()
    
plt.title("Structured Jet Model")
plt.axvline(thetaj, color = "k", linestyle = "-.", label = r"$\theta_j={:.3f}$".format(thetaj))
plt.axvline(THETA,  color = "k",  linestyle = ":", label = r"$\Theta={:.3f}$".format(THETA))
plt.plot(theta, jettophat(theta), color = c5, linewidth = 3, linestyle = "--", label = "top hat")
plt.plot(theta, jetstruct_plcutoff(theta), color = c6, linewidth = 3, label = "structured")
plt.legend()

plt.savefig("plots/jetStructure_plcutoff", dpi = 1000)
plt.savefig("plots/jetStructure_plcutoff.svg")