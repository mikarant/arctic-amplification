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
    axlist[1].set_ylim(0,0.5)
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

   
    meanratio = df_obs.loc[2019].mean()
    axlist[1].annotate('Observed average: '+str(meanratio.round(1)), (1.3,2.1), xycoords='data')
    axlist[1].axhline(y=meanratio, linestyle='--', linewidth=1.5)
    
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


def plot_aa_diff(df_anom, df_diff):
    
    window=20
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
    plt.plot(df_anom.rolling(window=window).mean(),linewidth=1)

    plt.plot(df_anom.mean(axis=1).rolling(window=window).mean(),'k' ,linewidth=2, label='CMIP6 mean')

    plt.plot(df_diff.mean(axis=1).rolling(window=window).mean(), '-',linewidth=3, color='r', zorder=5,label='Obs')
    
    
def trend_plot_2d(df_trends, df_err, df_slope_a, df_slope):
    
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
    # plt.scatter(df_trends.loc['Global trend']*10,df_trends.loc['Arctic trend']*10,c='k', s=40, label='Observations')
    plt.errorbar(df_trends.loc['Global trend']*10,df_trends.loc['Arctic trend']*10, 
                 yerr=df_err.loc['Arctic']*10, xerr=df_err.loc['Global']*10,
                 fmt='o', capsize=4, capthick=2, elinewidth=2, c='k',
                 label='Observations')

    plt.scatter(df_slope.loc[2019]*10,df_slope_a.loc[2019]*10,c='g', s=50, label='CMIP6 models')
    
    plt.xlabel('Global warming trend [K per decade]', fontsize=14)
    plt.ylabel('Arctic warming trend [K per decade]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.title('Temperature trends 1980-2019 globally and in the Arctic', fontsize=14)
    
    plt.legend(loc='upper left')
    
def sia_tas_scatter(df_slope_sic, df_slope_a, df_trends, slope_sic):
    
    fs = 14
    
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
    
    x = df_slope_sic.loc[2019].astype(float).values*1e-05/1000000
    y = df_slope_a.loc[2019].astype(float).values*10
    ind = np.isfinite(x)
    
    z = np.polyfit(x[ind], y[ind], 1)
    p = np.poly1d(z)
    
    x_obs = slope_sic*1e-05/1000000
    y_obs = df_trends.loc['Arctic trend'].mean()*10
    
    
   
    ax.scatter(x,y,c='g', s=50, label='CMIP6 models')
    ax.plot(x,p(x),"g--")
    ax.scatter(x_obs, y_obs, c='b')
    
    r = np.round(np.corrcoef(x[ind], y[ind])[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.95,0.9), xycoords='axes fraction',ha='right',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.95,0.83), xycoords='axes fraction',ha='right',
                fontsize=fs, c='b', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    plt.xlabel('Arctic sea ice area trend 1980-2019\n[million km² decade⁻¹]', fontsize=fs)
    plt.ylabel('Arctic temperature trend 1980-2019\n[K decade⁻¹]', fontsize=fs)


    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'tas_sia_trends.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
def sia_aa_scatter(df_slope_sic, df, df_obs, slope_sic):
    
    fs = 14
    
    
    ### years when the AA is THE largest
    years = df.loc[2010:2040].astype(float).idxmax()
    max_aa = df.loc[2010:2040].astype(float).max()
    
    xx = []
    ## find sea ice trend on those years
    for m in years.index:
        y = years[m]
        sic_trend = df_slope_sic[m][y]
        xx.append(sic_trend)
      
    # make the list as numpy array and convert m²/year to million km²/decade
    xx = np.array(xx)*1e-05/1000000
    
    # remove nans
    ind = np.isfinite(xx)
    xx = xx[ind]
    max_aa = max_aa[ind]
    
    # define observed values
    x_obs = slope_sic*1e-05/1000000
    y_obs = df_obs.loc[2018].mean()
    
    # figure
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
   
    sc = ax.scatter(xx,max_aa,c=years[ind], s=50, label='CMIP6 models')
    cbar = plt.colorbar(sc)    
    cbar.set_label('Year when the highest AA occur', rotation=90, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)


    ax.scatter(x_obs, y_obs, c='r')
    
    r = np.round(np.corrcoef(xx, max_aa)[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.05,0.1), xycoords='axes fraction',ha='left',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.05,0.05), xycoords='axes fraction',ha='left',
                fontsize=fs, c='r', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    plt.xlabel('40-year Arctic sea ice area trend \n[million km² decade⁻¹]', fontsize=fs)
    plt.ylabel('Maximum Arctic amplification ', fontsize=fs)


    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'tas_sia_aa.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    



