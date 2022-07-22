#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:02:41 2021

@author: rantanem
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

## fontsize
fs =14

obs_year=2021

# select probability time window for models
prob_years = np.arange(2012,2041)



#cmip6 models
models = pd.read_excel('/Users/rantanem/Documents/python/data/arctic_warming/list_of_models.xlsx')
models = models[models.ssp245>0]

cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_ann.csv', index_col=0)
cmip6_col = list(cmip6.columns)
canesm_col = [s for s in cmip6_col if 'CanESM5'.rstrip() in s]

cmip5 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa.csv', index_col=0)
cmip5_ratios_plot=cmip5.loc[prob_years].values.ravel()


# read MPI_GE model data
mpi_df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/mpi-ge/mpi-ge_aa_ann.csv', index_col=0)
mpi_ratios_plot=mpi_df.loc[prob_years].values.ravel()


# read CanESM5 model data
canesm_col = [s for s in cmip6_col if 'CanESM5'.rstrip() in s]
canesm_ratios_plot = cmip6[canesm_col].loc[prob_years].values.squeeze().ravel()


# exclude canesm5 from cmip6
cmip6_without_canesm = [m for m in cmip6_col if m not in canesm_col]
cmip6_all = cmip6.copy()
cmip6 = cmip6[cmip6_without_canesm]






df_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_ann.csv', index_col=0)



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
    
# calculate the probabilities for CMIP6 using only one realization per model
cmip6_r1 = pd.DataFrame(columns=models.Name, index=prob_years)
for m in models.Name:
    matching = [s for s in cmip6.columns if m.rstrip()+'_r1i1p1' in s]
    if m =='CESM2':
         matching = [s for s in cmip6.columns if m.rstrip()+'_r4i1p1' in s]
        
    if matching:
        ensemble = cmip6.loc[prob_years][matching[0]]
        cmip6_r1[m] = ensemble

cmip6_r1_plot = cmip6_r1.values.squeeze().ravel()


hist_aa = [cmip5_ratios_plot, cmip6_r1_plot, mpi_ratios_plot, canesm_ratios_plot]





titles = ['a) CMIP5', 'b) CMIP6','c) MPI-GE', 'd) CanESM5']

fig, axlist= plt.subplots(nrows=1, ncols=4, figsize=(14,4), dpi=200, sharey=True, sharex=False)

i=0
for ax in axlist:
    ax.hist(hist_aa[i], bins=np.arange(-0.5,6,0.25), density=True, facecolor='darkgrey', edgecolor='k')
    print(np.std(hist_aa[i]))
    ax.set_ylim(0,1.)
    ax.axvline(x=obs_aa, color='r' ,linewidth=2, label='Observations')
    

    cond = np.sum(hist_aa[i] >= obs_aa)
    prob = np.round(cond/len(hist_aa[i])*100,1)
    n = cond
    
    # if the probability is zero, use "almost equal" sign
    if prob < 0.001:
        sign = '≈'
    else:
        sign = '='
    ax.annotate('p '+sign+' '+str(prob)+' %', (-0.3,0.9))
    # ax.annotate('n = '+str(n)+' realizations', (-0.3,0.82))
    ax.set_title(titles[i], loc='left')
    ax.set_xlabel('AA', fontsize=fs)
    ax.tick_params(axis='both', which='major', labelsize=fs)
    i+=1
axlist[0].set_ylabel('Density', fontsize=fs)


figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'aa_pdfs_all_cmip6_r1.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


