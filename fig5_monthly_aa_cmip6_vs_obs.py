#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 08:23:37 2022

This script plots seasonal AA comparison with CMIP6 and observations
AA values are calculated 
@author: rantanem
"""

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from scipy import stats
import seaborn as sns
sns.set_theme()

# define whether include CanESM5 to CMIP6 or not
include_canesm5=False

months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov', 'dec', 'ann']

cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_ann.csv', index_col=0)
cmip6_col = list(cmip6.columns)
if not include_canesm5:
    print('\nCanESM5 not included!\n')
    canesm_col = [s for s in cmip6_col if 'CanESM5'.rstrip() in s]
    cmip6_col = [m for m in cmip6_col if m not in canesm_col]

# year
year = 2021

cmip6_aa_monthly = pd.DataFrame(index=cmip6_col, columns=months)
obs_aa_monthly = pd.DataFrame(index=['BEST','GISTEMP','HADCRUT','ERA5'], columns=months)
obs_perc_rank = pd.DataFrame(columns=months, index=['rank'])

for m in months:
    
    path_to_cmip6_file = '/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_seasonality/cmip6_aa_'+m+'.csv'
    file_exists = exists(path_to_cmip6_file)
    # if monthly file does not exist, use annual file
    if not file_exists:
        path_to_cmip6_file = '/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_ann.csv'
    
    
    df_aa = pd.read_csv(path_to_cmip6_file, index_col=0)[cmip6_col]
    
    cmip6_aa = df_aa.loc[year]
    print('Ensemble size for '+m+' '+str(len(cmip6_aa)))
    
    cmip6_aa_monthly[m] = cmip6_aa
    
    
    path_to_obs_file = '/Users/rantanem/Documents/python/data/arctic_warming/obs_aa_seasonality/observed_aa_'+m+'.csv'
    file_exists = exists(path_to_obs_file)
    if not file_exists:
        path_to_obs_file = '/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_ann.csv'

    df_obs = pd.read_csv(path_to_obs_file, index_col=0)

    obs_aa = df_obs.loc[year]

    obs_aa_monthly[m] = obs_aa
    
    obs_perc_rank[m] = stats.percentileofscore(cmip6_aa, obs_aa.mean())
    


fig = plt.figure(figsize=(9,6),dpi=200); ax1=plt.gca()

for i, m in enumerate(months[:12]):

    bp = plt.boxplot(cmip6_aa_monthly[m], positions=[i], widths=0.6,patch_artist=False, notch=False, whis=(5,95), sym='')
    
    sc = ax1.scatter(i, obs_aa_monthly[m].mean(), color='red', s=60, zorder=3)
    
    ax1.annotate(str(int(obs_perc_rank[m].round(0).values.squeeze()))+'%', (i*1/12+0.05, 1.01), 
                 xycoords='axes fraction', ha='center', fontsize=15)
    
ax1.grid(True, linestyle=':')  
plt.ylabel('Arctic amplification', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=16)
plt.ylim(0.,6.)


plt.xticks(np.arange(0,12), [each_string.capitalize() for each_string in months[:12]])

ann_ax = fig.add_axes([0.93, 0.125, 0.08, 0.755])
bp = ann_ax.boxplot(cmip6_aa_monthly['ann'], positions=[0], widths=0.6,patch_artist=False, notch=False, whis=(5,95), sym='')
plt.xticks([0], [months[-1].capitalize()])
plt.ylim(0.,6)
ann_ax.grid(True, linestyle=':') 
ann_ax.axes.yaxis.set_ticklabels([])
ann_ax.tick_params(axis='y',length=0)
ann_ax.tick_params(axis='both', which='major', labelsize=16)

sc = ann_ax.scatter([0], obs_aa_monthly['ann'].mean(), color='red', s=60, zorder=3, label='Observations')


ann_ax.annotate(str(int(obs_perc_rank['ann'].round(0).values.squeeze()))+'%', (0.6, 1.01), 
             xycoords='axes fraction', ha='center', fontsize=15)

ann_ax.legend(bbox_to_anchor=((-0.34,0.12)), fontsize=16)

plt.savefig('/Users/rantanem/Documents/python/figures/figure5.pdf',dpi=300,bbox_inches='tight')
