#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 08:49:17 2021

@author: rantanem
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

### read observed
df_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends.csv', index_col=0)
df_err = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_errors.csv', index_col=0)
df_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa.csv', index_col=0)

### read cmip5 results
cmip5_ref_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_trends_ref.csv',index_col=0)
cmip5_arctic_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_trends_arctic.csv',index_col=0)


### read cmip6 results
cmip6_ref_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_global_trends.csv',index_col=0)
cmip6_arctic_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_arctic_trends.csv',index_col=0)

## read canesm5 results
canesm_col = [s for s in cmip6_ref_trends.columns if 'CanESM5'.rstrip() in s]
canesm_arctic_trends = cmip6_arctic_trends[canesm_col]
canesm_ref_trends = cmip6_ref_trends[canesm_col]

## read mpi results
mpi_ds = xr.open_dataset('/Users/rantanem/Documents/python/data/arctic_warming/data_pdf_plots_MPI-ESM_rcp45.nc')



# exclude canesm5 from cmip6
cmip6_without_canesm = [m for m in cmip6_ref_trends.columns  if m not in canesm_col]
cmip6_ref_trends = cmip6_ref_trends[cmip6_without_canesm]
cmip6_arctic_trends = cmip6_arctic_trends[cmip6_without_canesm]


longnames = ['CMIP5',
             'CMIP6',
             'MPI-GE',
             'CanESM5',
             ]
titles = ['a)','b)','c)','d)',]

cmap = plt.get_cmap("tab10")

## select year
year=2018
mpi_ind =  np.isin(np.arange(1889,2100), year)

# select 1980-2019 period

x1 = cmip5_ref_trends.loc[year]*10
y1 = cmip5_arctic_trends.loc[year]*10
x2 = cmip6_ref_trends.loc[year]*10
y2 = cmip6_arctic_trends.loc[year]*10
x3 = mpi_ds['global_trend'].values[:,mpi_ind]
y3 = mpi_ds['arctic_trend'].values[:,mpi_ind]
x4 = canesm_ref_trends.loc[year]*10
y4 = canesm_arctic_trends.loc[year]*10


x_scatters = [x1, x2, x3, x4]
y_scatters = [y1, y2, y3, y4]

# calculate linear trend
ind = np.isfinite(x1)
z = np.polyfit(x1[ind], y1[ind], 1)
p = np.poly1d(z)
px = np.linspace(0.1, 0.5, 100)

dTa = df_obs.loc[year].mean()*px




fig, axlist= plt.subplots(nrows=2, ncols=2, figsize=(10,8), dpi=200, sharey=True, sharex=False)
axlist=axlist.ravel()

i=0
for ax in axlist:
    ax.scatter(x_scatters[i], y_scatters[i], s=30, color=cmap(i), zorder=5)
    ax.errorbar(df_trends.loc['Global trend']*10,df_trends.loc['Arctic trend']*10, 
                 yerr=df_err.loc['Arctic']*10, xerr=df_err.loc['Global']*10,
                 fmt='o', capsize=4, capthick=1, elinewidth=1, c='k',
                 label='Observations', zorder=6)
    ax.plot(px, dTa, 'k--')
    ax.set_xlim(0.1,0.5)
    ax.set_ylim(-0.05,1.8)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    ax.annotate(longnames[i], (0.05,0.9), xycoords='axes fraction', fontsize=16,
                fontweight='bold', color=cmap(i))
    ax.annotate(longnames[i], (0.05,0.9), xycoords='axes fraction', fontsize=16,
                fontweight='bold', color=cmap(i))
    ax.annotate(titles[i], (-0.15,0.95), xycoords='axes fraction', fontsize=20,
                fontweight='bold', color='k')
    ax.grid(True)
    i+=1

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.ylabel("Arctic warming trend [K per decade]\n", fontsize=18)
plt.xlabel("\nGlobal warming trend [K per decade]", fontsize=18)

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'global_warming_vs_arctic_warming.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')