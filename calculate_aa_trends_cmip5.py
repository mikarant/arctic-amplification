#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:32:42 2020
This script calculates Arctic amplification ratios for CMIP5 models

@author: rantanem
"""
import xarray as xr
import numpy as np
import pandas as pd
from cdo import *
cdo = Cdo()
from scipy import stats
import seaborn
seaborn.reset_orig()


### lenght of period (default = 40 years)
period = 40

# starting year (the first year which is included for the linear trends)
startYear = 1961

# latitude threshold for the Arctic
latitude_threshold = 66.5

# reference area ('global', 'nh' or 'sh')
refarea = 'global'

# Season ('DJF' etc. For annual, select 'ANN')
season = 'ANN'

## choose cmip5 or cmip6 models
modelGeneration = 'cmip5'


# choose the scenario
ssp='rcp45'

if modelGeneration == 'cmip5':
    ssp = 'rcp85'


# variable
var = 'tas'

# open model dataset
path = '/Users/rantanem/Documents/python/data/arctic_warming/cmip5/'+ssp+'.nc'
cmip5 = xr.open_dataset(path)


if type(season) == int:
    operator = '-selmon,'
    season = str(season)
elif type(season) == str:
    operator = '-selseason,'

years = np.arange(startYear,2100-period+1)

mod = list(cmip5['source_id'].values)

# allocate dataframes
df =pd.DataFrame(index=years+period-1, columns=mod)
df_slope =pd.DataFrame(index=years+period-1, columns=mod)
df_slope_a =pd.DataFrame(index=years+period-1, columns=mod)
temp_arctic = pd.DataFrame(index=np.unique(cmip5.year), columns=mod)
temp = pd.DataFrame(index=np.unique(cmip5.year), columns=mod)

# loop over all models
for m in mod[:]:
    print('Calculating '+m)
    path = '/Users/rantanem/Documents/python/data/arctic_warming/cmip5/'+m+'_cmip5.nc'
    
    # calculate annual means
    yearmean = cdo.yearmean(input=operator+season+' '+ path)
    # regrid to 0.5° grid using CDO conservative remapping function
    rm  = cdo.remapcon('r720x360 -selvar,tas ',input=yearmean, options='-b F32') 
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
        ratio = slope_a/slope
        df[m][y+period-1] = ratio
        df_slope[m][y+period-1] = slope
        df_slope_a[m][y+period-1] = slope_a


        
        
## print global mean and arctic mean trends ffrom  models
print('Global mean trend 1980-2019:')
print(str(np.round(df_slope.loc[2019].mean(),4)))
print('Arctic mean trend 1980-2019:')
print(str(np.round(df_slope_a.loc[2019].mean(),4)))
print('Arctic amplification 1980-2019:')
print(str(np.round(df.loc[2019].mean(),4)))

# output the data
df_slope.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_global_trends.csv')
df_slope_a.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_arctic_trends.csv')
df.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa.csv')

