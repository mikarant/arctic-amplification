#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:18:42 2021

@author: rantanem
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




def plot_trends(df_trends, df_err, df_slope_a, df_slope, season, annot):

    df_slope_a = df_slope_a.astype(float)
    df_slope = df_slope.astype(float)
    cmap = plt.get_cmap("tab10")
    # long names of observations
    longnames = ['Berkeley\nEarth',
                 'Gistemp',
                 'Cowtan&\nWay',
                 'ERA5',
                 ]
    range_min = 0.05
    range_max = 0.95

    xticks = np.arange(1,5)
    fig, axlist= plt.subplots(nrows=2, ncols=1, figsize=(9,7), dpi=200, sharex=True)
    axlist[0].errorbar(xticks-0.05, df_trends.loc['Arctic trend']*10, yerr=1.6448*df_err.loc['Arctic']*10, 
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0), label='Arctic trend')
    if annot:
        i=1
        for o in df_trends.columns:
            y = df_trends.loc['Arctic trend',o]*10
            err = 1.6448*df_err.loc['Arctic',o]*10
            ymax = y + err
            ymin = y - err
            axlist[0].annotate(str(np.round(y,3)) + '\n± '+str(np.round(err,3)), 
                               (i,y-0.07),)
            i+=1
    
    axlist[1].errorbar(xticks-0.05, df_trends.loc['Global trend']*10, yerr=1.6448*df_err.loc['Global']*10,
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(1), label='Global trend')
    
    if annot:
        i=1
        for o in df_trends.columns:
            y = df_trends.loc['Global trend',o]*10
            err = 1.6448*df_err.loc['Global',o]*10
            ymax = y + err
            ymin = y - err
            axlist[1].annotate(str(np.round(y,3)) + '\n± '+str(np.round(err,3)), 
                               (i+0.02,y-0.02),)
            i+=1

    ## plot cmip6
    y = df_slope_a.loc[2019].mean(axis=0)*10
    ymin = y - df_slope_a.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope_a.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    
    axlist[0].errorbar(4.95,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))
    
    
    y = df_slope.loc[2019].mean(axis=0)*10
    ymin = y - df_slope.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    axlist[1].errorbar(5.05, y, yerr=yerr,
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(1))


    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText("Arctic",loc='upper left', prop=dict(size=14), frameon=True)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axlist[0].add_artist(at)
    at = AnchoredText("Global",loc='upper left', prop=dict(size=14), frameon=True)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axlist[1].add_artist(at)

    axlist[0].grid(axis='y')
    axlist[1].grid(axis='y')
    axlist[0].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)
    axlist[1].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)
    axlist[0].set_ylim(0,1.4)
    axlist[1].set_ylim(0,0.4)
    axlist[0].tick_params(axis='both', which='major', labelsize=14)
    axlist[1].tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(np.append(xticks,(5)),labels=longnames  + ['CMIP6\nmean'], fontsize=14)
    axlist[0].set_title('Temperature trends in '+season+ ' season', fontsize=18)

    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'trends_obs_vs_models.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
def plot_trends_aa(df_trends, df_err, df_slope_a, df_slope, df_obs, df_obs_min, df_obs_max, df, season, annot):
    
    df_slope_a = df_slope_a.astype(float)
    df_slope = df_slope.astype(float)
    df = df.astype(float)
    cmap = plt.get_cmap("tab10")
    # long names of observations
    longnames = ['Berkeley\nEarth',
                 'Gistemp',
                 'Cowtan&\nWay',
                 'ERA5',
                 ]
    
    range_min = 0.05
    range_max = 0.95
    
    ### read cmip5 results
    cmip5_ref_trends = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_trends_ref.csv',index_col=0)
    cmip5_arctic_trends = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_trends_arctic.csv',index_col=0)
    cmip5_ratios = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_ratios.csv',index_col=0)
    
    xticks = np.arange(1,5)

    fig, axlist= plt.subplots(nrows=2, ncols=1, figsize=(9,7), dpi=200, sharex=True)

    axlist[0].errorbar(xticks-0.05, df_trends.loc['Arctic trend']*10, yerr=1.6448*df_err.loc['Arctic']*10, 
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0), label='Arctic trend')
    axlist[0].errorbar(xticks+0.05, df_trends.loc['Global trend']*10, yerr=1.6448*df_err.loc['Global']*10,
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(1), label='Global trend')
    ## plot cmip5
    axlist[0].errorbar(4.95, cmip5_ref_trends.loc[2019].mean()*10, yerr=1.6448*cmip5_ref_trends.loc[2019].std()*10,
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(1))
    axlist[0].errorbar(5.05, cmip5_arctic_trends.loc[2019].mean()*10, yerr=1.6448*cmip5_arctic_trends.loc[2019].std()*10,
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))
    
    ## plot cmip6 for arctic
    y = df_slope_a.loc[2019].mean(axis=0)*10
    ymin = y - df_slope_a.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope_a.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(6.05,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))
    
    ## plot cmip6 for global
    y = df_slope.loc[2019].mean(axis=0)*10
    ymin = y - df_slope.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(5.95, y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(1))


    axlist[0].grid(axis='y')
    axlist[0].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)

    from matplotlib import container
    handles, labels = axlist[0].get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

    axlist[0].legend(handles, labels, fontsize=14, ncol=2, loc='upper left')

    plt.xticks(np.append(xticks,(5,6)),labels=longnames + ['CMIP5\nmean'] + ['CMIP6\nmean'], fontsize=14)
    axlist[0].set_yticks(np.arange(0,1.3,0.2))
    axlist[0].tick_params(axis='both', which='major', labelsize=14)
    axlist[0].tick_params(axis='both', which='minor', labelsize=14)


    ymin = df_obs.loc[2019] - df_obs_min.loc[2019]
    ymax = df_obs_max.loc[2019] - df_obs.loc[2019]


    xticks = np.arange(1,5)

    axlist[1].errorbar(xticks, df_obs.loc[2019], yerr=(ymin,ymax), 
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))
    
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

    
    axlist[1].errorbar(5, cmip5_ratios.loc[2019].mean(), yerr=1.6448*cmip5_ratios.loc[2019].std(),
                        fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))

    meanratio = df_obs.loc[2019].mean().round(2)
    axlist[1].annotate('Observed average: '+str(meanratio), (1.3,2.1), xycoords='data')
    
    ## plot cmip6
    y = df.loc[2019].mean()
    ymin = y - df.quantile(range_min, axis=1)[2019]
    ymax =  df.quantile(range_max, axis=1)[2019] - y
    yerr = [[ymin],[ymax]]
    
    axlist[1].errorbar(6, y, yerr=yerr,
                       fmt='o', capsize=5, capthick=2, elinewidth=3, c=cmap(0))


    axlist[1].grid(axis='y')
    axlist[1].set_ylabel('Arctic amplification', fontsize=14)

    plt.xticks(np.append(xticks,(5,6)),labels=longnames + ['CMIP5\nmean'] + ['CMIP6\nmean'], fontsize=14)
    plt.yticks(np.arange(1.0,5,0.5), fontsize=14)

    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'aa_obs_vs_models.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')

