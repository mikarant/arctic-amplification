#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:32:42 2020
This script calculates Arctic amplification ratios for CMIP5 models

@author: rantanem
"""
import xarray as xr
import glob
import os
import numpy as np
import pandas as pd
import pymannkendall as mk 
from cdo import *
cdo = Cdo()
from scipy import stats
import datetime
import seaborn
seaborn.reset_orig()


### lenght of period (default = 40 years)
period = 43

# starting year (the first year which is included for the linear trends)
startYear = 1950

# latitude threshold for the Arctic
latitude_threshold = 66.5

# reference area ('global', 'nh' or 'sh')
refarea = 'global'

# Season ('DJF' etc. For annual, select 'ANN')
season = 'ANN'

if season=='ANN':
    month_name=season.lower()
else:
    month_name = datetime.date(1900, season, 1).strftime('%b').lower()

# choose the scenario (only rcp4.5 available currently)
ssp='rcp45'

# variable
var = 'tas'

# determine month name
if type(season) == int:
    operator = '-selmon,'
    season = str(season)
elif type(season) == str:
    operator = '-selseason,'

years = np.arange(startYear,2100-period+1)

model_paths = glob.glob('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/'+ssp+'/merged/*.nc')
model_filenames = [i.split('/')[-1] for i in model_paths]
models = [i.split('.')[0] for i in model_filenames]

models_df = pd.DataFrame(data=sorted(models, key=str.lower), index=np.arange(1,len(models)+1), columns=['Model'])

# allocate dataframes
df =pd.DataFrame(index=years+period-1, columns=models)
df_slope =pd.DataFrame(index=years+period-1, columns=models)
df_slope_a =pd.DataFrame(index=years+period-1, columns=models)
temp_arctic = pd.DataFrame(index=np.arange(1901, 2100), columns=models)
temp = pd.DataFrame(index=np.arange(1901, 2100), columns=models)

# loop over all models
for m in models:
    print('Calculating '+m)
    path = '/Users/rantanem/Documents/python/data/arctic_warming/cmip5/'+ssp+'/merged/'+m+'.nc'
    
    
    # calculate annual means
    yearmean = cdo.yearmean(input=operator+season+' '+ path)
    # regrid to 0.5° grid using CDO conservative remapping function
    rm  = cdo.remapcon('r720x360 -selyear,1901/2099 -selvar,tas ',input=yearmean, options='-b F32') 
    ds = xr.open_dataset(rm)
    
    
    
    if refarea =='global':
        cond = ds.lat>=-90
    elif refarea =='nh':
        cond = ds.lat>=0
    elif refarea =='sh':
        cond = ds.lat<=0
    
    # weights for the model grid
    weights = np.cos(np.deg2rad(ds.lat))

    
    # select global temperature
    t = ds[var].where(cond).weighted(weights).mean(("lon", "lat")).squeeze()
    clim = t.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    temp[m] = t - clim
    
    # select Arctic temperature
    t_a = ds[var].where(ds.lat>=latitude_threshold).weighted(weights).mean(("lon", "lat")).squeeze()
    clim = t_a.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    temp_arctic[m] = t_a - clim
    
    # calculate trends and AA ratios
    for y in years:
        yrange = np.arange(y,y+period)
        f = temp[m][yrange]
        f_a = temp_arctic[m][yrange]
        slope, _, _, p_value, _ = stats.linregress(yrange, f.values)
        slope_a, _, _, p_value_a, _ = stats.linregress(yrange, f_a.values)
        
        result = mk.original_test(f.values, alpha=0.05)
        
        # constrain AA by global warming rate
        #if the global warming trend is not statistically singificant according to MK test
        if result.trend=='increasing':
            ratio = slope_a/slope
        else:
            ratio = np.nan
        
        df[m][y+period-1] = ratio
        df_slope[m][y+period-1] = slope
        df_slope_a[m][y+period-1] = slope_a
    
    os.remove(rm)
    os.remove(yearmean)


        
        
## print global mean and arctic mean trends from  models
print('Global mean trend 1979-2021:')
print(str(np.round(df_slope.loc[2021].mean(),4)))
print('Arctic mean trend 1979-2021:')
print(str(np.round(df_slope_a.loc[2021].mean(),4)))
print('Arctic amplification 1979-2021:')
print(str(np.round(df.loc[2021].mean(),4)))

# output the data
df_slope.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_global_trends_'+month_name+'.csv',
                index_label='Year', na_rep='NaN')
df_slope_a.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_arctic_trends_'+month_name+'.csv',
                  index_label='Year', na_rep='NaN')
df.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa_'+month_name+'.csv',
          index_label='Year', na_rep='NaN')

