# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:37:36 2025

@author: rohdo
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import transforms

c1 = "#553CA5"
c2 = "#006EB2"
c3 = "#96B4DF"
c4 = "#F0E442"
c5 = "#E69F00"
c6 = "#D55E00"
c7 = "#750000"

def logslope(x, y):
    """"""
    x = x.astype(float)
    y = y.astype(float)
    return np.gradient(np.log(y), np.log(x))

def mostFrequent(x, a, M = 5):
    """np.unique preserving order of first instance and dropping terms with 
    count less than given M.
    """
    x_unique, counts = np.unique(np.round(x, a), return_counts = True)
    x_unique, indices = np.unique(np.round(x, a), return_index = True)
    x_unique = np.where(counts <= M, np.nan, x_unique)
    indices = np.where(counts <= M, np.nan, indices)
    x_unique = x_unique[~np.isnan(x_unique)]
    indices = indices[~np.isnan(indices)]
    x_unique = x_unique[np.argsort(indices)]#[:len(x_unique)])] # FIXME
    
    return x_unique

def alpha(x, y, a = 3):
    """"""
    return mostFrequent(logslope(x, y), a)

def nFormatStr(n):
    """"""
    s = ""
    
    for i in range(n - 1):
        s += "{}, "
        
    s += "{}"
    
    return s

def plotrect(xmin, xmax, ymin, ymax, n = 100, **kwargs):
    """"""
    x = np.linspace(xmin, xmax, n)
    
    y1 = np.ones_like(x) * ymin
    y2 = np.ones_like(x) * ymax
    
    plt.fill_between(x, y1, y2 = y2, **kwargs)

def plotlogfan(x0, y0, a, aerr, R, color = "k", res = 100, r = None, label = None, linewidth = 1, zorder = 1e50):
    """"""
    f = lambda x, a : y0 * (x/x0)**a
    
    a_1std = np.geomspace(a - aerr, a + aerr, res) # TODO maybe should be linspace?
    
    for i, A in enumerate(a_1std[:-1]):
        xmax = x0 * 10**(np.sqrt(R**2/(1 + A**2)))
        
        if r is None:
            xmin = x0
        else:
            xmin = x0 * 10**(np.sqrt(r**2/(1 + A**2)))
        
        xcont = np.geomspace(xmin, xmax, 10)
        Af = a_1std[i + 1]
        
        if i == 0:
            l = label
        else:
            l = None
        
        plt.fill_between(xcont, f(xcont, A), f(xcont, Af), color = color, label = l, zorder = zorder, linewidth = linewidth)
        
    xmax_a = x0 * 10**(np.sqrt(R**2/(1 + a**2)))
    ymax_a = f(xmax, a)
    
    return (xmax_a, ymax_a)

def pvalue(yobs, yobserr, y, yerr):
    """"""
    ydiff = abs(yobs - y)
    ydifferr = np.sqrt(yobserr**2 + yerr**2)
    
    p = scipy.stats.norm.cdf(0, ydiff, ydifferr)
    
    return p

def nsigma(yobs, yobserr, y, yerr):
    """"""
    diff = abs(yobs - y)
    
    if yobserr != 0:
        nsigma1 = diff/yobserr
    else:
        nsigma1 = np.inf
    
    if yerr != 0:
        nsigma2 = diff/yerr
    else:
        nsigma2 = np.inf
    
    return nsigma1, nsigma2

def plotErrorComparison(title, ylabel, fname, *args, scale = "linear", dpi = 1000, figsize = (4, 8), top = None, bottom = None, elinewidth = 4, labelsize = 10, titlesize = 20, ylabelsize = 16, legend = True, markerscale = 0.5, legendsize = 14):
    """"""
    plt.figure(figsize = figsize)
    plt.title(title, fontsize = titlesize)
    plt.yscale(scale)
    plt.ylabel(ylabel, fontsize = ylabelsize)
    plt.xlim(-1, len(args))
    
    labels = []
    xloc = []
    
    yobs = args[0][0]
    yobserr = args[0][1]
    
    for x, arg in enumerate(args):
        y = arg[0]
        yerr = arg[1]
        label = arg[2]
        color = arg[3]

        plt.axhline(y, linestyle = "-", color = color, alpha = 0.5, linewidth = elinewidth/1.5)
        plotline, _, _ = plt.errorbar(x, y, yerr = yerr, fmt = "o", color = color,\
                                      markeredgecolor = color, ms = 5 * elinewidth,\
                                      mew = elinewidth/1.2, elinewidth = elinewidth,\
                                      label = label)
        plotline.set_markerfacecolor('none')
        
        xloc.append(x)
        
        if x != 0:
            p = pvalue(yobs, yobserr, y, yerr)
            nsigma1, nsigma2 = nsigma(yobs, yobserr, y, yerr)
            
            if nsigma1 != np.inf:
                nsigma1str = "\n{:.1f}".format(nsigma1) + r"$\sigma_{obs}$"
            else:
                nsigma1str = ""
            
            if nsigma2 != np.inf:
                nsigma2str = "\n{:.1f}".format(nsigma2) + r"$\sigma_{mod}$"
            else:
                nsigma2str = ""
            
            if p == 0:
                pstr = "\np=0"
            elif p <= 0.001:
                pstr = "\np={:.1e}".format(p)
            else:
                pstr = "\np={:.2f}".format(p)
            
            labels.append(label + "\n" + pstr + \
                          nsigma1str +\
                          nsigma2str)
        
        else:
            labels.append(label)
        
    plt.xticks(xloc, labels, fontsize = labelsize)
    plt.yticks(fontsize = labelsize)
    
    if legend:
        plt.legend(markerscale = markerscale, fontsize = legendsize, loc = "lower left")
    
    if not(top is None):
        plt.ylim(top = top)
    if not(bottom is None):
        plt.ylim(bottom = bottom)    
    
    plt.tight_layout()
    
    plt.savefig(fname, dpi = dpi)
    plt.savefig(fname + ".svg")
    plt.close()

def multicolored_text(x, y, text, cd, **kw): # TODO does not work with maplotlib tex
    """
    https://stackoverflow.com/questions/9169052/partial-coloring-of-text
    Place text at (x, y) with colors for each word specified in the color
    dictionary cd, which maps strings to colors, and must include a 'default'
    key mapping to the default color.

    Based on https://stackoverflow.com/a/9185851/2683, thanks paul ivanov!
    """
    fig = plt.gcf()
    t = plt.gca().transData

    def get_text_width(text, **kw):
        temp = plt.text(0, 0, text, **kw)
        temp.draw(fig.canvas.get_renderer())
        ex = temp.get_window_extent()
        # Need the text width in data coordinates, since that's what x uses.
        width = t.inverted().transform_bbox(ex).width
        temp.remove()
        return width

    ha = kw.pop('ha', 'left')
    
    if ha == 'left':
        x = x
    elif ha == 'center':
        x -= get_text_width(text, **kw) / 2
    elif ha == 'right':
        x -= get_text_width(text, **kw)
    else:
        raise ValueError(f'invalid value for horizontal alignment {ha}')

    for word in text.split(' '):
        c = cd.get(word, cd['default'])
        text = plt.text(x, y, word + ' ', color=c, transform=t, **kw)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        # Need the text width in points so that this will work with pdf output.
        width = ex.width / fig.dpi * 72  # Can do via fig.dpi_scale_trans?

        t = transforms.offset_copy(text._transform, x=width, units='points', fig=fig)
    
