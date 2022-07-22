#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 07:04:20 2022

@author: rantanem
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize

Reds = cm.get_cmap('Reds', 12)
Blues = cm.get_cmap('Blues', 10)
newcolors = Reds(np.linspace(0, 1, 12))
newcolors[0, :] = Blues(0.8)
newcolors[1, :] = Blues(0.2)
for ii in range(2, 12, 2):
    newcolors[ii, :] = newcolors[ii + 1, :]
newcolors = np.vstack((newcolors, newcolors[-1], newcolors[-1]))
newcolors[-2:, :3] = newcolors[-2:, :3] * 0.4
newcm = ListedColormap(newcolors)
norm2D = Normalize(vmin=0, vmax=7)

model='cmip6'

Ncolors = 20
cmdiff = ListedColormap(plt.get_cmap('RdYlBu_r', Ncolors)(np.linspace(0, 1, Ncolors)[1:-1]))
normdiff = Normalize(5, 95)
cmdiff.set_over(plt.get_cmap('RdYlBu_r', Ncolors)(np.linspace(0, 1, Ncolors)[-1]))
cmdiff.set_under(plt.get_cmap('RdYlBu_r', Ncolors)(np.linspace(0, 1, Ncolors)[0]))


aa_df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/obs_aa_sensitivity.csv', index_col=0)


perc_df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/obs_aa_perc_'+model+'.csv', index_col=0)


latitudes = aa_df.index
time_window = np.array(aa_df.columns.astype(float))

aa_df = aa_df.astype(float)
perc_df = perc_df.astype(float)

fig, ax= plt.subplots(nrows=1, ncols=2, figsize=(14,5.5), dpi=300)
fs=16


sns.heatmap(aa_df.round(1).astype(float), cmap=newcm, norm=norm2D, ax=ax[0], annot=True, fmt=".3g",
            yticklabels=2, cbar_kws={'label': 'Arctic amplification', 'extend':'max', })


sns.heatmap(perc_df.round(0).astype(int), cmap=cmdiff, norm=normdiff, ax=ax[1],
            yticklabels=2, annot=True, fmt=".3g",
            cbar_kws={'label': 'Percentile rank', 'extend':'both', 'ticks':np.arange(10,100,10)})

x_idx = np.interp(43, time_window-2.5, range(len(time_window-2.5)))
y_idx = np.interp(66.5, latitudes-1.25, range(len(latitudes)))


for i in np.arange(0, len(ax)):
    ax[i].scatter(x_idx, y_idx, marker='*', s=120, zorder=10, color='yellow')
    ax[i].set_yticklabels(latitudes[::2].astype(int), fontsize=14, va='center')
    ax[i].set_xlabel('Time window (years)', fontsize=fs)
    ax[i].set_ylabel('Arctic region latitude threshold (°)', labelpad=12, fontsize=fs)
    ax[i].invert_yaxis()
    ax[i].tick_params(axis='both', which='major', labelsize=16)
    cbar = ax[i].collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.yaxis.label.set_size(fs)

ax[0].set_title('a)', fontsize=fs)

ax[1].set_title('b)', fontsize=fs)

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'figure2.pdf'
  
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')