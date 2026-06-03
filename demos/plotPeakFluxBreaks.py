# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:08:45 2026

@author: rohdo
"""

from rsjetstruct.rsjetstruct import RSjetStruct
import numpy as np
import matplotlib.pyplot as plt
import os

tobs = np.geomspace(3e-4, 1e-2, 1000)

model = RSjetStruct(tobs, 1e10, 5.787037e-04, 1e1, 2e10, 1e13, 7e9, keps = 0, kGamma = 0,\
             k = 0, p = 2.5, g = 5, tjet = np.inf, weigthed = True)
    
base_dir = os.path.dirname(__file__)          # .../rsjetstruct/demos
project_dir = os.path.dirname(base_dir)        # .../rsjetstruct

out_dir = os.path.join(project_dir, "plots")
os.makedirs(out_dir, exist_ok=True)
    
plt.xlabel("Time (days)")
plt.ylabel("Flux density (mJy) (before SSA)")
plt.loglog(tobs, model.Fnumaxrs(), color = "k", linestyle = "-")

out_path = os.path.join(out_dir, "TemplateMaxFlux.png")
out_path_svg = os.path.join(out_dir, "TemplateMaxFlux.svg")
plt.savefig(out_path, dpi = 500)
plt.savefig(out_path_svg)

plt.cla()

plt.xlabel("Time (days)")
plt.ylabel(r"$\nu_a$ (Hz)")
plt.loglog(tobs, model.nuars(), color = "k", linestyle = "-")

out_path = os.path.join(out_dir, "TemplateNuars.png")
out_path_svg = os.path.join(out_dir, "TemplateNuars.svg")
plt.savefig(out_path, dpi = 500)
plt.savefig(out_path_svg)

