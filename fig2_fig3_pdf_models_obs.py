#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:02:41 2021

@author: rantanem
"""
from collections import Counter
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

## fontsize
fs =14

# select observed AA year
obs_year=2018

# select probability time window for models
prob_years = np.arange(2010,2041)

# select time series  window for models
timser_years = np.arange(2000,2041)


# select the time periods from MPI-GE data
mpi_ind_timeser = np.isin(np.arange(1889,2100), timser_years)
mpi_ind_hist = np.isin(np.arange(1889,2100), prob_years)


# cmip6 model names
models = pd.read_excel('/Users/rantanem/Documents/python/data/arctic_warming/list_of_models.xlsx')
models = models[models.ssp245>0]


cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)
cmip6_col = list(cmip6.columns)

# calculate the number of realizations from each model
all_models = [word for line in cmip6_col for word in line.split('_')][::2]
number_of_members = Counter(all_models)


cmip5 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa.csv', index_col=0)
cmip5_ratios_plot=cmip5.loc[prob_years].values.ravel()

# read MPI_GE model data
mpi_ds = xr.open_dataset('/Users/rantanem/Documents/python/data/arctic_warming/data_pdf_plots_MPI-ESM_rcp45.nc')
mpi_timeser = pd.DataFrame(data=mpi_ds.aa.values[:,mpi_ind_timeser].transpose(), index=np.arange(2000,2041), columns=np.arange(1,101))
mpi_ratios_plot=mpi_ds.aa.values[:,mpi_ind_hist].squeeze().ravel()

# read CanESM5 model data
canesm_col = [s for s in cmip6_col if 'CanESM5'.rstrip() in s]
canesm_ratios_plot = cmip6[canesm_col].loc[prob_years].values.squeeze().ravel()


# exclude canesm5 from cmip6
cmip6_without_canesm = [m for m in cmip6_col if m not in canesm_col]
cmip6_all = cmip6.copy()
cmip6 = cmip6[cmip6_without_canesm]
cmip6_all_ratios_plot=cmip6.loc[prob_years].values.ravel()


# read observed AA
df_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa.csv', index_col=0)
obs_aa = df_obs.loc[obs_year].mean(axis=0)



### calculate the probabilities for CMIP6
## each model has weight of 1
cmip6_probs = pd.DataFrame(index=models.Name, columns=['n','p', 'n_aa'])
for m in models.Name:
    matching = [s for s in cmip6.columns if m.rstrip() in s]
    ensemble = cmip6.loc[prob_years][matching]
    cond = np.sum(ensemble >= obs_aa).sum()
    prob_ensemble = np.round(cond/np.size(ensemble)*100,1)
    cmip6_probs.loc[m].n = len(matching)
    cmip6_probs.loc[m].p = prob_ensemble
    cmip6_probs.loc[m].n_aa = cond
    


### PLOT PDF diagrams
# distributions for the datasets
hist_aa = [cmip5_ratios_plot, cmip6_all_ratios_plot, mpi_ratios_plot, canesm_ratios_plot]

titles = ['a) CMIP5', 'b) CMIP6','c) MPI-GE', 'd) CanESM5']

fig, axlist= plt.subplots(nrows=1, ncols=4, figsize=(14,4), dpi=200, sharey=True, sharex=False)

i=0
for ax in axlist:
    print(titles[i])
    # plot the histograms
    ax.hist(hist_aa[i], bins=np.arange(-0.5,6,0.25), density=True, facecolor='darkgrey', edgecolor='k')
    
    ax.set_ylim(0,1.)
    #plot the observed AA
    ax.axvline(x=obs_aa, color='r' ,linewidth=2, label='Observations')
    
    # replace CMIP6 probability with weight 1 for each model
    if i==1:
        prob = np.round(cmip6_probs.p.mean(),1)
        n = cmip6_probs.n_aa.sum()
    else:
        cond = np.sum(hist_aa[i] >= obs_aa)
        prob = np.round(cond/len(hist_aa[i])*100,1)
        n = cond
    
    # if the probability is zero, use "almost equal" sign
    if prob < 0.001:
        sign = '≈'
    else:
        sign = '='
    
    # annotate p-value
    ax.annotate('p '+sign+' '+str(prob)+' %', (-0.3,0.9))

    ax.set_title(titles[i], loc='left')
    ax.set_xlabel('AA', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    i+=1

axlist[0].set_ylabel('Density', fontsize=fs)

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'aa_pdfs_all.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


### PLOT time series

fig, ax= plt.subplots(nrows=1, ncols=4, figsize=(14,4), dpi=200, sharex=False)

plt.subplots_adjust(wspace=0.25) 

ax[0].plot(cmip5.index, cmip5, linewidth=0.5, color='grey')
ax[0].plot(cmip5.index, cmip5.mean(axis=1), linewidth=1.5, color='k')
obs_plot = ax[0].plot(df_obs.index, df_obs.mean(axis=1), color='red', label='Observations')


ax[1].plot(cmip6.index, cmip6, linewidth=0.5, color='grey')
ax[1].plot(cmip6.index, cmip6.mean(axis=1), linewidth=1.5, color='k')
ax[1].plot(df_obs.index, df_obs.mean(axis=1), color='red')


ax[2].plot(mpi_timeser.index, mpi_timeser, linewidth=0.5, color='grey')
ax[2].plot(mpi_timeser.index, np.mean(mpi_timeser,axis=1), linewidth=1.5, color='k')
ax[2].plot(df_obs.index, df_obs.mean(axis=1), color='red', label='Observations')


ax[3].plot(cmip6.index, cmip6_all[canesm_col], linewidth=0.5, color='grey')
ax[3].plot(cmip6.index, cmip6_all[canesm_col].mean(axis=1), linewidth=1.5, color='k')
ax[3].plot(df_obs.index, df_obs.mean(axis=1), color='red')

# modify the axis parameters
for i in range(0,4):
    ax[i].set_ylim(-1,5.5)
    ax[i].set_xlim(2000,2040)
    ax[i].tick_params(axis='both', which='major', labelsize=fs)
    ax[i].set_title(titles[i],loc='left',  fontsize=fs)
    ax[i].axvspan(2000, 2010, facecolor='grey', alpha=0.45, lw=0, linewidth=0.5)



ax[0].legend(handles=obs_plot, loc='upper center', bbox_to_anchor=(0.55, 1.02),
                     edgecolor='none', ncol=3, fontsize=fs)
ax[0].set_ylabel('Arctic amplification', fontsize=fs)

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'aa_timeser_all.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
 


