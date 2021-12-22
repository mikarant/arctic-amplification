#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:31:09 2021

This script reads global and arctic area-mean temperatures and calculates
trends and AA ratios for CMIP6 models

@author: mprantan
"""
import numpy as np
import pandas as pd
import xarray as xr
import glob
from scipy import stats
import sys
import pymannkendall as mk

# input path for the data
path = '/users/mprantan/python/arctic/data/'

# define the period for which the AA and trends are calculated
period = 40

years = np.arange(1961, 2100-period +1)

# input data
df_global = pd.read_csv(path + 'cmip6_global_temps.csv', index_col=0)
df_arctic = pd.read_csv(path + 'cmip6_arctic_temps.csv', index_col=0)


simulations = df_global.columns

#allocate lists
models_global_trend_list = []
models_arctic_trend_list = []
models_aa_list = []

# loop over all realizations
for m in simulations[:]:
    print(m)
    
    
    # global and arctic temperatures
    t = df_global[m]
    t_a = df_arctic[m]
    
    df_global_trend = pd.DataFrame(index=years+period-1, columns=[m])
    df_arctic_trend = pd.DataFrame(index=years+period-1, columns=[m])
    df_aa = pd.DataFrame(index=years+period-1, columns=[m])
    
    # loop over all 40-year periods
    for y in years:
        yrange = np.arange(y, y+period)
        
        # 40-year global and arctic temperatures
        f = t.loc[yrange]
        f_a = t_a.loc[yrange]
        
        # calculate temperature trends and their p-values
        slope, _, _, p_value, _ = stats.linregress(yrange, f.values)
        slope_a, _, _, p_value_a, _ = stats.linregress(yrange, f_a.values)
        
        # test whether the trend is increasing
        result = mk.original_test(f.values, alpha=0.05)
        
        # constrain AA by global warming rate
        #if the global warming trend is not statistically singificant according to MK test
        if result.trend=='increasing':
            ratio = slope_a/slope
        else:
            ratio = np.nan
            print('NO TREND')

        # store the trends and AA ratios
        df_global_trend.loc[y+period-1] = slope
        df_arctic_trend.loc[y+period-1] = slope_a
        df_aa.loc[y+period-1] = ratio
    
    # append them to lists
    models_global_trend_list.append(df_global_trend)
    models_arctic_trend_list.append(df_arctic_trend)
    models_aa_list.append(df_aa)
    
# convert lists to dataframes
models_global_trend_df = pd.concat(models_global_trend_list, axis=1)
models_arctic_trend_df = pd.concat(models_arctic_trend_list, axis=1)
models_aa_df = pd.concat(models_aa_list, axis=1)

# output the dataframes as csv files
models_global_trend_df.to_csv(path+'cmip6_global_trends.csv')
models_arctic_trend_df.to_csv(path+'cmip6_arctic_trends.csv')
models_aa_df.to_csv(path+'cmip6_aa.csv')
        
        
