#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 16:03:19 2021

@author: rantanem
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy import stats
import pymannkendall as mk 
from cdo import *
cdo = Cdo()

def get_sia_trend(sia_file, varname, period, factor):
    
    ds = xr.open_dataset(sia_file)
    obs = ds[varname].sel(time=slice(str(period[0])+"-01-01", str(period[1])+"-12-31")).squeeze()*factor
    clim = obs.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    obs_anom = obs-clim
    
    idx = np.isfinite(obs_anom.values)
    slope_sia, _, _, p_value_sia, stderr = stats.linregress(np.arange(period[0],period[1]+1)[idx], obs_anom.values[idx])
    
    return slope_sia, obs.values, stderr

def get_sia_area(sia_file, varname, factor):
    
    ds = xr.open_dataset(sia_file)
    obs = ds[varname].sel(time=slice("1979-01-01", "2018-12-31")).squeeze()*factor

    idx = np.isfinite(obs.values)

    mean_sia = np.mean(obs.values[idx])
    
    return mean_sia

def getRatioPresent(temp_ref, temp_arctic, obsname, yrange, period):
    
    f = temp_ref[obsname][yrange]
    f_a = temp_arctic[obsname][yrange]
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f.values)
    slope_a, _, r_value_a, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    
    result = mk.original_test(f.values, alpha=0.05)
    
    # constrain AA by global warming rate
    #if the global warming trend is not statistically singificant according to MK test
    if result.trend=='increasing':
        ratio = slope_a/slope
    else:
        ratio = np.nan
        print('NO TREND')
    
    return ratio, r_value_a
    
    

def getRatioObs(temp_ref, temp_arctic, obsname, yrange, period):

    import pymannkendall as mk
    

    Nsamples = 100000

    
    f = temp_ref[obsname][yrange]
    f_a = temp_arctic[obsname][yrange]
    
    result = mk.original_test(f,alpha=0.05)
    result_a = mk.original_test(f_a,alpha=0.05)
    
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f.values)
    slope_a, _, _, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    ratio = slope_a/slope
    
    if (result.trend != 'increasing') or (result_a.trend != 'increasing'):
        ratio = np.nan
    
    slope_arctic_samples = slope_a + stderr_a * np.random.randn(Nsamples)
    slope_global_samples = slope + stderr * np.random.randn(Nsamples)
    
    confidence_int_min = np.percentile(slope_arctic_samples / slope_global_samples, 5)
    confidence_int_max = np.percentile(slope_arctic_samples / slope_global_samples, 95)
    
    return ratio, confidence_int_min, confidence_int_max

def getObsTemps(obsname, varname, latitude_threshold,refarea, operator, season):
 

    # variables in observations
    filenames = {'GISTEMP': '/Users/rantanem/Documents/python/data/arctic_warming/GISTEMP-regridded.nc',
                 'BEST': '/Users/rantanem/Documents/python/data/arctic_warming/BEST-regridded-retimed.nc',
                 'HADCRUT': '/Users/rantanem/Documents/python/data/arctic_warming/hadcrut5-regridded.nc',
                 'ERA5': '/Users/rantanem/Documents/python/data/arctic_warming/era5_t2m_1950-2021.nc',
                 }

    inputfile = filenames[obsname]
    file = cdo.yearmean(input=operator+season+' '+inputfile)

    ds = xr.open_dataset(file)


    # define coordinate names
    latname = (ds['latitude'] if 'latitude' in ds else ds['lat']).name
    lonname = (ds['longitude'] if 'longitude' in ds else ds['lon']).name

    # define weights  
    weights = np.cos(np.deg2rad(ds[latname]))

    if refarea =='global':
        cond = ds[latname]>=-90
    elif refarea =='nh':
        cond = ds[latname]>=0
    elif refarea =='sh':
        cond = ds[latname]<=0    


    # calculate temperatures
    temp_ref = ds[varname].where(cond).weighted(weights).mean((lonname, latname)).squeeze()
    clim = temp_ref.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    anom_ref = temp_ref - clim
    
    temp_arctic = ds[varname].where(ds[latname]>=latitude_threshold).weighted(weights).mean((lonname, latname)).squeeze()
    clim = temp_arctic.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    anom_arctic = temp_arctic - clim

    index = pd.to_datetime(ds.time.values).year

    
    df = pd.DataFrame(index=index, columns=['Reference temperature', 'Arctic temperature'])
    
    df['Reference temperature'] = anom_ref
    df['Arctic temperature'] = anom_arctic
    

    return df