#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:03:07 2022

@author: rantanem
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()



# read observed AA
aa_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_to_present.csv', index_col=0)

# read CMIP5 AA
cmip5 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa_to_present.csv', index_col=0)

# read MPI-GE AA
mpi = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/mpi-ge/mpi-ge_aa_to_present.csv', index_col=0)


# read cmip6 AA
cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_to_present.csv', index_col=0)
cmip6_col = list(cmip6.columns)

# read CanESM5 model data
canesm_col = [s for s in cmip6_col if 'CanESM5'.rstrip() in s]
canesm = cmip6[canesm_col]#.loc[prob_years].values.squeeze().ravel()


# exclude canesm5 from cmip6
cmip6_without_canesm = [m for m in cmip6_col if m not in canesm_col]
cmip6_all = cmip6.copy()
cmip6 = cmip6[cmip6_without_canesm]




### PLOT time series

## fontsize
fs =14


# titles
titles = ['a) CMIP5', 'b) CMIP6','c) MPI-GE', 'd) CanESM5']

fig, ax= plt.subplots(nrows=1, ncols=4, figsize=(14,4), dpi=200, sharex=False)

plt.subplots_adjust(wspace=0.25) 


for c in cmip5.columns:
    ax[0].plot(cmip5.index, cmip5[c], linewidth=0.5, color='grey')
ax[0].plot(cmip5.index, cmip5.mean(axis=1), linewidth=1.5, color='k')
obs_plot = ax[0].plot(aa_obs.index, aa_obs.mean(axis=1), color='red', label='Observations')
ax[0].axvline(1979, linestyle=':')

for c in cmip6.columns:
    ax[1].plot(cmip6.index, cmip6[c], linewidth=0.5, color='grey')
ax[1].plot(cmip6.index, cmip6.mean(axis=1), linewidth=1.5, color='k')
ax[1].plot(aa_obs.index, aa_obs.mean(axis=1), color='red')
ax[1].axvline(1979, linestyle=':')

for c in mpi.columns:
    ax[2].plot(mpi.index, mpi[c], linewidth=0.5, color='grey')
ax[2].plot(mpi.index, mpi.mean(axis=1), linewidth=1.5, color='k')
ax[2].plot(aa_obs.index, aa_obs.mean(axis=1), color='red')
ax[2].axvline(1979, linestyle=':')

for c in cmip6_all[canesm_col].columns:
    ax[3].plot(cmip6.index, cmip6_all[canesm_col][c], linewidth=0.5, color='grey')
ax[3].plot(cmip6.index, cmip6_all[canesm_col].mean(axis=1), linewidth=1.5, color='k')
ax[3].plot(aa_obs.index, aa_obs.mean(axis=1), color='red')
ax[3].axvline(1979, linestyle=':')

# modify the axis parameters
for i in range(0,4):
    ax[i].set_ylim(-1.,6)
    ax[i].set_xlim(1950,2012)
    ax[i].set_xticks(np.arange(1950,2023,20))
    ax[i].tick_params(axis='both', which='major', labelsize=fs)
    ax[i].set_title(titles[i],loc='left',  fontsize=fs)
    



ax[0].legend(handles=obs_plot, loc='upper center', bbox_to_anchor=(0.43, 0.15),
                     edgecolor='none', ncol=3, fontsize=fs)
ax[0].set_ylabel('Arctic amplification', fontsize=fs)

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'figure4.pdf'
  
plt.savefig(figurePath + figureName,dpi=300,bbox_inches='tight')