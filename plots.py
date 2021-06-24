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

    fig, axlist= plt.subplots(nrows=2, ncols=1, figsize=(9,7), dpi=200, sharex=False)

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

    axlist[0].set_yticks(np.arange(0,1.3,0.2))
    axlist[0].tick_params(axis='both', which='major', labelsize=14)
    axlist[0].tick_params(axis='both', which='minor', labelsize=14)


    ymin = df_obs.loc[2019] - df_obs_min.loc[2019]
    ymax = df_obs_max.loc[2019] - df_obs.loc[2019]

    axlist[0].set_xticks(np.append(xticks,(5,6)))
    axlist[0].set_xticklabels(labels=longnames + ['CMIP5\nmean'] + ['CMIP6\nmean'], fontsize=14)


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

    plt.subplots_adjust(hspace=0.3)

    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'aa_obs_vs_models.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')

    
def trend_plot_2d(df_trends, df_err, df_slope_a, df_slope):
    
    x = df_slope.loc[2019].astype(float).values*10
    y = df_slope_a.loc[2019].astype(float).values*10
    ind = np.isfinite(x)
    
    z = np.polyfit(x[ind], y[ind], 1)
    p = np.poly1d(z)
    px = np.linspace(0.1, 0.45, 100)
    
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
    # plt.scatter(df_trends.loc['Global trend']*10,df_trends.loc['Arctic trend']*10,c='k', s=40, label='Observations')
    plt.errorbar(df_trends.loc['Global trend']*10,df_trends.loc['Arctic trend']*10, 
                 yerr=df_err.loc['Arctic']*10, xerr=df_err.loc['Global']*10,
                 fmt='o', capsize=4, capthick=1, elinewidth=1, c='k',
                 label='Observations')

    plt.scatter(df_slope.loc[2019]*10,df_slope_a.loc[2019]*10,c='g', s=50, label='CMIP6 models')
    ax.plot(px,p(px),"g--")
    
    plt.xlabel('Global warming trend [K per decade]', fontsize=14)
    plt.ylabel('Arctic warming trend [K per decade]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    plt.title('Temperature trends 1980-2019 globally and in the Arctic', fontsize=14)
    
    plt.legend(loc='upper left')
    
def sia_tas_scatter(df_slope_sic, df_slope_a, df_trends, sia_slopes, sia_trend_errs, df_err):
    
    fs = 14
    

    plt.figure(figsize=(7.2,6.08), dpi=200)
    ax=plt.gca()
    
    x = df_slope_sic.loc[2019].astype(float).values*1e-05/1000000
    y = df_slope_a.loc[2019].astype(float).values*10
    ind = np.isfinite(x)
    
    z = np.polyfit(x[ind], y[ind], 1)
    p = np.poly1d(z)
    px = np.linspace(-1.25, 0.05, 100)
    
    x_obs = sia_slopes*1e-05/1000000
    y_obs = df_trends.loc['Arctic trend'].mean()*10
    
    # sigma_tot_x=np.sqrt(np.sum((sia_trend_errs.astype(float)*1e-05/1000000)**2))
    # sigma_tot_y=np.sqrt(np.sum((df_err.loc['Arctic'].astype(float)*10)**2))
   
    sigma_tot_x=np.mean(sia_trend_errs.astype(float)*1e-05/1000000)
    sigma_tot_y=np.mean(df_err.loc['Arctic'].astype(float)*10)
    
   
    ax.scatter(x,y,c='g', s=50, label='CMIP6 models', edgecolor='k', linewidth=0.5)
    ax.plot(px,p(px),"g--")
    # for o in x_obs.index:
    #     ax.scatter(x_obs.loc[o], y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    
    ax.scatter(x_obs.mean(), y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    ax.errorbar(x_obs.mean(), y_obs, xerr=1.645*sigma_tot_x, yerr=1.645*sigma_tot_y, capsize=4, capthick=1, elinewidth=1, 
                fmt='o', c='r')

    
    r = np.round(np.corrcoef(x[ind], y[ind])[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.95,0.9), xycoords='axes fraction',ha='right',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.95,0.83), xycoords='axes fraction',ha='right',
                fontsize=fs, c='r', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    plt.xlabel('Arctic sea ice area trend 1980-2019\n[million km² decade⁻¹]', fontsize=fs)
    plt.ylabel('Arctic temperature trend 1980-2019\n[K decade⁻¹]', fontsize=fs)
    ax.annotate('a)', xy=(0.48, 0.93), xycoords='figure fraction', fontsize=fs+4, fontweight='bold')


    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'tas_sia_trends.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
def sia_aa_scatter(df_slope_sic, df, df_obs, sia_slopes, df_slope, df_trends, sia_trend_errs):
    
    ### read errorbars for maximum AA
    max_err = pd.read_csv('/home/rantanem/Documents/python/arctic-amplification/bootstrapCI_temps_obs_19792018.csv',index_col=0 )
    max_err_min = max_err['ratio'] - max_err['CIlowerPercentile']
    max_err_max = max_err['CIupperPercentile'] - max_err['ratio']
    
    yerr = [[np.mean(max_err_max)],[np.mean(max_err_min)]]
    
    sia_errs = sia_trend_errs / df_trends.loc['Global trend'].mean()
    sigma_tot_x=np.mean(sia_errs.astype(float))/(1000000*1000000)
    
    
    fs = 14
    # d = df.drop(columns=['UKESM1-0-LL', 'EC-Earth3'])
    d = df
    ### years when the AA is THE largest
    years = d.loc[2010:2040].astype(float).idxmax()
    max_aa = d.loc[2010:2040].astype(float).max()
    
    models = list(years.index)
    xx = []
    ## find sea ice trend on those years
    for m in models:
        y = years[m]
        sic_trend = df_slope_sic[m][y] / df_slope[m][y]
        xx.append(sic_trend)
      
    # make the list as numpy array and convert m²/year to million km²/decade
    xx = np.array(xx)/(1000*1000*1000000)
    
    # remove nans
    ind = np.isfinite(xx)
    xx = xx[ind]
    max_aa = max_aa[ind]
    
    z = np.polyfit(xx, max_aa, 1)
    p = np.poly1d(z)
    px = np.linspace(-3.5, -0.75, 100)
    
    
    # define observed values
    x_obs = (sia_slopes/ df_trends.loc['Global trend'].mean())/(1000*1000*1000000)#*1e-05/1000000
    # x_obs2 = (slope_sic2/ df_trends.loc['Global trend'].mean())/(1000*1000*1000000)#*1e-05/1000000
    y_obs = df_obs.loc[2018].mean()
    
    # figure
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
   
    sc = ax.scatter(xx,max_aa,c=years[ind], s=50, label='CMIP6 models', cmap='viridis',
                    edgecolor='k', linewidth=0.5)
    cbar = plt.colorbar(sc)    
    cbar.set_label('Year when the highest AA occur', rotation=90, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)
    ax.plot(px,p(px),"g--")

    # for o in x_obs.index:
    #     ax.scatter(x_obs.loc[o], y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    # ax.scatter(x_obs.mean(), y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    
    ax.errorbar(x_obs.mean(), y_obs, yerr=yerr, xerr=1.645*sigma_tot_x, capsize=4, capthick=1, elinewidth=1, 
                fmt='o', c='r')
    
    
    r = np.round(np.corrcoef(xx, max_aa)[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.05,0.1), xycoords='axes fraction',ha='left',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.05,0.05), xycoords='axes fraction',ha='left',
                fontsize=fs, c='r', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    plt.xlabel('40-year dSIA/dGMST \n[million km²/°C]', fontsize=fs)
    plt.ylabel('Maximum Arctic amplification ', fontsize=fs)
    ax.annotate('b)', xy=(0.45, 0.93), xycoords='figure fraction', fontsize=fs+4, fontweight='bold')

    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'tas_sia_aa.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
def meansia_aa_scatter(df, df_slope_sic, df_obs, mean_sic, sia_areas, sia_standard_errors):
    
    fs = 14
    
    
    ### years when the AA is THE largest
    years = df.loc[2010:2040].astype(float).idxmax()
    max_aa = df.loc[2010:2040].astype(float).max()
    
    xx = []
    ## find sea ice trend on those years
    for m in years.index:
        y = years[m]
        sic_trend = np.mean(mean_sic.loc[y-39:y][m])
        sic_trend = np.mean(mean_sic.loc[1980:2019][m])

        # sic_trend = np.mean(df_slope_sic.loc[y-39:y][m])
        xx.append(sic_trend)
    
    # make the list as numpy array and convert m²/year to million km²/decade
    xx = np.array(xx)/(1000*1000*1000000)
    
    # remove nans
    ind = np.isfinite(xx)
    xx = xx[ind]
    max_aa = max_aa[ind]
    
    ### mean sea ice area values
    mean_sia_models = mean_sic.loc[1980:2019].mean()/(1000*1000*1000000)

    
    ### trend values
    mean_sia_trends_models = df_slope_sic.loc[2019]*1e-05/1000000
    mean_sia_obs = sia_areas/(1000*1000*1000000)
    
    sigma_tot=np.sqrt(np.sum((sia_standard_errors.astype(float)/(1000*1000*1000000))**2))

    # remove nans
    ind = np.isfinite(mean_sia_models)
    mean_sia_models = mean_sia_models[ind]
    # max_aa=max_aa[ind]
    # years =years[ind]
    mean_sia_trends_models = mean_sia_trends_models[ind].astype(float)
    
    # define observed values
    y_obs = df_obs.loc[2018].mean()

    
    # figure
    plt.figure(figsize=(9,6), dpi=200)
    ax=plt.gca()
   
    # ax.scatter(mean_sia_models,mean_sia_trends_models,c='g', s=50, label='CMIP6 models')
    sc = ax.scatter(xx,max_aa,c=years[ind], s=50, label='CMIP6 models', edgecolor='k', linewidth=0.5)
    cbar = plt.colorbar(sc)    
    cbar.set_label('Year when the highest AA occur', rotation=90, fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)

    # for o in mean_sia_obs.index:
    #     ax.scatter(mean_sia_obs.loc[o], y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    # ax.scatter(mean_sia_obs.mean(), y_obs, c='r',  s=50, edgecolor='k', linewidth=0.5)
    ax.errorbar(mean_sia_obs.mean(), y_obs, xerr=1.645*sigma_tot, capsize=4, capthick=1, elinewidth=1, 
                fmt='o', c='r')
    
    r = np.round(np.corrcoef(xx, max_aa)[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.05,0.1), xycoords='axes fraction',ha='left',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.05,0.05), xycoords='axes fraction',ha='left',
                fontsize=fs, c='r', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    # plt.xlabel('40-year Arctic mean sea ice area \n[million km²]', fontsize=fs)
    # plt.ylabel('40-year Arctic sea ice area trend \n[million km² decade⁻¹]', fontsize=fs)
    plt.xlabel('40-year Arctic mean sea ice area \n[million km²]', fontsize=fs)
    plt.ylabel('Maximum Arctic amplification', fontsize=fs)
    ax.annotate('c)', xy=(0.45, 0.93), xycoords='figure fraction', fontsize=fs+4, fontweight='bold')


    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'max_aa_vs_mean_sia.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
def meansia_siatrend_scatter(df, df_slope_sic, df_obs, mean_sic, sic_obs, slope_sic):

    fs = 14
    
       
    ### mean sea ice area values
    mean_sia_models = mean_sic.loc[1980:2019].mean()/(1000*1000*1000000)

    
    ### trend values
    mean_sia_trends_models = df_slope_sic.loc[2019]*1e-05/1000000
    mean_sia_obs = sic_obs.sel(time=slice("1980-01-01", "2019-12-31")).mean().values/(1000*1000*1000000)
    

    # remove nans
    ind = np.isfinite(mean_sia_models)
    mean_sia_models = mean_sia_models[ind]
    # max_aa=max_aa[ind]
    # years =years[ind]
    mean_sia_trends_models = mean_sia_trends_models[ind].astype(float)
    
    # define observed values
    y_obs = slope_sic*1e-05/1000000
    # y_obs = df_obs.loc[2018].mean()

    
    # figure
    plt.figure(figsize=(7.2,6.08), dpi=200)
    ax=plt.gca()
   
    ax.scatter(mean_sia_models,mean_sia_trends_models,c='g', s=50, label='CMIP6 models')
    # sc = ax.scatter(xx,max_aa,c=years[ind], s=50, label='CMIP6 models')
    # cbar = plt.colorbar(sc)    
    # cbar.set_label('Year when the highest AA occur', rotation=90, fontsize=fs)
    # cbar.ax.tick_params(labelsize=fs)


    ax.scatter(mean_sia_obs, y_obs, c='r')
    
    r = np.round(np.corrcoef(mean_sia_models, mean_sia_trends_models)[0][1],2)
    
    ax.annotate('CMIP6 models: R= '+str(r), xy=(0.05,0.1), xycoords='axes fraction',ha='left',
                fontsize=fs, c='g', fontweight='bold')
    ax.annotate('Observed', xy=(0.05,0.05), xycoords='axes fraction',ha='left',
                fontsize=fs, c='r', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.tick_params(axis='both', which='minor', labelsize=fs)
    plt.xlabel('40-year Arctic mean sea ice area \n[million km²]', fontsize=fs)
    plt.ylabel('40-year Arctic sea ice area trend \n[million km² decade⁻¹]', fontsize=fs)
    # plt.xlabel('40-year Arctic mean sea ice area \n[million km²]', fontsize=fs)
    # plt.ylabel('Maximum Arctic amplification', fontsize=fs)
    ax.annotate('c)', xy=(0.5, 0.93), xycoords='figure fraction', fontsize=fs+4, fontweight='bold')


    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'mean_sia_trend_vs_mean_sia.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')
    
    

def plot_fig_4(df_trends, df_err, df_slope_a, df_slope, df_obs, df_obs_min, df_obs_max, df, annot):
    
    df_slope_a = df_slope_a.astype(float)
    df_slope = df_slope.astype(float)
    df = df.astype(float)
    cmap = plt.get_cmap("tab10")
    # long names of observations
    longnames = ['Berkeley\nEarth',
                 'Gistemp',
                 'Cowtan &\nWay',
                 'ERA5',
                 ]
    
    range_min = 0.05
    range_max = 0.95
    
    elinewidth = 2.5
    
    ### read cmip5 results
    cmip5_ref_trends = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_trends_ref.csv',index_col=0)
    cmip5_arctic_trends = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_trends_arctic.csv',index_col=0)
    cmip5_ratios = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_ratios.csv',index_col=0)
    
    ### read mpi results
    mpi_aa = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/data_for_fig4.csv')
    
    ### read errorbars for maximum
    max_err = pd.read_csv('/home/rantanem/Documents/python/arctic-amplification/bootstrapCI_temps_obs_19792018.csv',index_col=0 )
    max_err_min = max_err['ratio'] - max_err['CIlowerPercentile']
    max_err_max = max_err['CIupperPercentile'] - max_err['ratio'] 

    # calculate cmip5 & cmip6 maximum 
    syear=2010
    eyear=2040
    df_max = pd.DataFrame(df.loc[syear:eyear].max(), columns=['aa'], index=df.loc[2019].index)
    df_max_cmip5 = pd.DataFrame(cmip5_ratios.loc[syear:eyear].max(), columns=['aa'], index=cmip5_ratios.loc[2019].index)

    bmin = np.mean(df_max.values) - np.quantile(df_max,range_min)
    bmax =   np.quantile(df_max,range_max)- np.mean(df_max.values)
    berr = [[bmin],[bmax]]
    
    cmin = np.mean(df_max_cmip5.values) - np.quantile(df_max_cmip5,range_min)
    cmax =   np.quantile(df_max_cmip5,range_max)- np.mean(df_max_cmip5.values)
    cerr = [[bmin],[bmax]]



    xticks = np.arange(1,5)

    fig, axlist= plt.subplots(nrows=2, ncols=1, figsize=(9,7), dpi=200, sharex=False)

    axlist[0].errorbar(xticks-0.05, df_trends.loc['Arctic trend']*10, yerr=1.6448*df_err.loc['Arctic']*10, 
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0), label='Arctic trend')
    axlist[0].errorbar(xticks+0.05, df_trends.loc['Global trend']*10, yerr=1.6448*df_err.loc['Global']*10,
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1), label='Global trend')
    
    # plot cmip5 for arctic
    y = cmip5_arctic_trends.loc[2019].mean(axis=0)*10
    ymin = y - cmip5_arctic_trends.quantile(range_min, axis=1)[2019]*10
    ymax =  cmip5_arctic_trends.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(4.95,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
    ## plot cmip5 for global
    y = cmip5_ref_trends.loc[2019].mean(axis=0)*10
    ymin = y - cmip5_ref_trends.quantile(range_min, axis=1)[2019]*10
    ymax =  cmip5_ref_trends.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(5.05, y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))
    
    ## plot cmip6 for arctic
    y = df_slope_a.loc[2019].mean(axis=0)*10
    ymin = y - df_slope_a.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope_a.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(5.95,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
    ## plot cmip6 for global
    y = df_slope.loc[2019].mean(axis=0)*10
    ymin = y - df_slope.quantile(range_min, axis=1)[2019]*10
    ymax =  df_slope.quantile(range_max, axis=1)[2019]*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(6.05, y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))

    ## plot mpi-ge for arctic
    y = mpi_aa['Arctic trend'].mean(axis=0)*10
    ymin = y - mpi_aa['Arctic trend'].quantile(range_min)*10
    ymax =  mpi_aa['Arctic trend'].quantile(range_max,)*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(6.95,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
     ## plot mpi-ge for global
    y = mpi_aa['Global trend'].mean(axis=0)*10
    ymin = y - mpi_aa['Global trend'].quantile(range_min)*10
    ymax =  mpi_aa['Global trend'].quantile(range_max,)*10 - y
    yerr = [[ymin],[ymax]]
    
    axlist[0].errorbar(7.05,y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))


    axlist[0].grid(axis='y')
    axlist[0].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)


    from matplotlib import container
    handles, labels = axlist[0].get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

    axlist[0].legend(handles, labels, fontsize=12, ncol=2, loc='upper left')

    axlist[0].set_xticks(np.append(xticks,(5,6,7)))
    axlist[0].set_xticklabels(labels=longnames + ['CMIP5'] + ['CMIP6'] + ['MPI-GE'], fontsize=12)
    axlist[0].set_yticks(np.arange(0,1.3,0.2))
    axlist[0].tick_params(axis='y', which='major', labelsize=14)


    ymin = df_obs.loc[2019] - df_obs_min.loc[2019]
    ymax = df_obs_max.loc[2019] - df_obs.loc[2019]


    xticks = np.arange(1,5)-0.07

    axlist[1].errorbar(xticks, df_obs.loc[2019], yerr=(ymin,ymax), 
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
    xticks = np.arange(1,5)+0.07
    
    

    axlist[1].errorbar(xticks, df_obs.loc[2018], yerr=(max_err_min,max_err_max), 
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))
    
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

    


    ## plot cmip5
    y = cmip5_ratios.loc[2019].mean()
    ymin = y - cmip5_ratios.quantile(range_min, axis=1)[2019]
    ymax =  cmip5_ratios.quantile(range_max, axis=1)[2019] - y
    yerr = [[ymin],[ymax]]


    axlist[1].errorbar(4.93, y, yerr=yerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
    axlist[1].errorbar(5.07, df_max_cmip5.mean(), yerr=cerr,
                        fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))

   
    # meanratio = df_obs.loc[2019].mean()
    # axlist[1].annotate('Observed average: '+str(meanratio.round(1)), (1.3,2.1), xycoords='data')
    # axlist[1].axhline(y=meanratio, linestyle='--', linewidth=1.5)
    
    ## plot cmip6
    y = df.loc[2019].mean()
    ymin = y - df.quantile(range_min, axis=1)[2019]
    ymax =  df.quantile(range_max, axis=1)[2019] - y
    yerr = [[ymin],[ymax]]
    
    axlist[1].errorbar(5.93, y, yerr=yerr,
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

    ## plot cmip6 maximum
    y = df_max.mean().squeeze()

    axlist[1].errorbar(6.07, y, yerr=berr,
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))



    ## plot mpi mean & maximum
    y = mpi_aa.AA.mean()
    ymin = y - mpi_aa.AA.quantile(range_min)
    ymax =  mpi_aa.AA.quantile(range_max) - y
    yerr = [[ymin],[ymax]]
    
    axlist[1].errorbar(6.93, y, yerr=yerr,
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))
    
    y = mpi_aa.AA_max.mean()
    ymin = y - mpi_aa.AA_max.quantile(range_min)
    ymax =  mpi_aa.AA_max.quantile(range_max) - y
    yerr = [[ymin],[ymax]]
    
    axlist[1].errorbar(7.03, y, yerr=yerr,
                       fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))


    


    

    axlist[1].grid(axis='y')
    axlist[1].set_ylabel('Arctic amplification', fontsize=14)

    labels = ['1980-2019 mean', '2010-2040 maximum']
    axlist[1].legend(handles, labels, fontsize=12, ncol=2, loc='lower left')

    plt.xticks(np.append(xticks,(5,6,7)),labels=longnames + ['CMIP5'] + ['CMIP6']
               + ['MPI-GE'], fontsize=12)
    plt.yticks(np.arange(1.0,5,0.5), fontsize=14)

    # increase width between the plots
    plt.subplots_adjust(hspace=0.4)
    
    ## add a and b labels
    axlist[0].annotate('a)', xy=(-0.2, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold')
    axlist[1].annotate('b)', xy=(-0.2, 0.5), xycoords='axes fraction', fontsize=14, fontweight='bold')

    figurePath = '/home/rantanem/Documents/python/figures/'
    figureName = 'aa_obs_vs_models.png'
  
    plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


