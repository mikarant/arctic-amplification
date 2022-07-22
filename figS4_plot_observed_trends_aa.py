#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:48:56 2021

@author: rantanem
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends_1979-2021_ann.csv', index_col=0)
df_err = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_errors_1979-2021_ann.csv', index_col=0)
df_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_ann.csv', index_col=0)


#ending year of the trend and AA
year=2021


## annotations
annot=False


cmap = plt.get_cmap("tab10")
longnames = ['Berkeley\nEarth',
             'Gistemp',
             'HadCRUT5',
             'ERA5',
             ]

range_min = 0.05
range_max = 0.95

elinewidth = 1
capthick=1


### read errorbars for AA
max_err = pd.read_csv('/Users/rantanem/Documents/python/arctic-amplification/bootstrapCI_temps_obs_19792021.csv',index_col=0 )
max_err_min = max_err['ratio'] - max_err['CIlowerPercentile']
max_err_max = max_err['CIupperPercentile'] - max_err['ratio'] 



xticks = np.arange(1,5)

fig, axlist= plt.subplots(nrows=1, ncols=2, figsize=(9,4), dpi=200, sharex=False)

axlist[0].errorbar(xticks-0.05, df_trends.loc['Arctic trend']*10, yerr=1.6448*df_err.loc['Arctic']*10, 
                   fmt='o', capsize=5, capthick=capthick, elinewidth=elinewidth, c=cmap(0), label='Arctic trend')
axlist[0].errorbar(xticks+0.05, df_trends.loc['Global trend']*10, yerr=1.6448*df_err.loc['Global']*10,
                   fmt='o', capsize=5, capthick=capthick, elinewidth=elinewidth, c=cmap(1), label='Global trend')


axlist[0].grid(axis='y')
axlist[0].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)


from matplotlib import container
handles, labels = axlist[0].get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

axlist[0].legend(handles, labels, fontsize=12, ncol=2, bbox_to_anchor=(1.02, 1.15))

axlist[0].set_xticks(xticks)
axlist[0].set_xticklabels(labels=longnames , fontsize=12)
axlist[0].set_yticks(np.arange(0,1.3,0.2))
axlist[0].tick_params(axis='y', which='major', labelsize=14)
axlist[0].set_ylim(0.14,0.92)


ymin = max_err_min
ymax = max_err_max


xticks = np.arange(1,5)-0.00

axlist[1].errorbar(xticks, df_obs.loc[year], yerr=(ymin,ymax), 
                    fmt='o', capsize=5, capthick=capthick, elinewidth=elinewidth, c=cmap(0))



if annot:
    i=1
    for o in df_obs.columns:
        y = df_obs.loc[2019,o]
        # err = 1.6448*df_err.loc['Arctic',o]*10
        # ymax = y + err
        # ymin = y - err
        axlist[1].annotate(str(np.round(y,3)), 
                           (i+0.1,y-0.09),)
        i+=1









axlist[1].grid(axis='y')
axlist[1].set_ylabel('Arctic amplification', fontsize=14)

labels = ['AA 1979-2021',]
axlist[1].legend(handles, labels, fontsize=12, ncol=1, bbox_to_anchor=(1.02, 1.15))

plt.xticks(xticks,labels=longnames, fontsize=12)
plt.yticks(np.arange(1.0,5.5,0.5), fontsize=14)
axlist[1].set_ylim(3,5)

# increase width between the plots
plt.subplots_adjust(wspace=0.5)

## add a and b labels
axlist[0].annotate('a)', xy=(-0.40, 1), xycoords='axes fraction', fontsize=16, fontweight='bold')
axlist[1].annotate('b)', xy=(-0.34, 1), xycoords='axes fraction', fontsize=16, fontweight='bold')

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'aa_obs.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')