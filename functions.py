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



def getRatioObs(temp_ref, temp_arctic, obsname, yrange, period):

    f = temp_ref[obsname][yrange]
    f_a = temp_arctic[obsname][yrange]
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f.values)
    slope_a, _, _, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    ratio = slope_a/slope

    return ratio 

def getObsTemps(obsname, varname, latitude_threshold,refarea):
    
    # open dataset
    ds = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/'+obsname+'_annual.nc')

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
    temp_arctic = ds[varname].where(ds[latname]>=latitude_threshold).weighted(weights).mean((lonname, latname)).squeeze()
    
    dtype = ds.time.values.dtype
    
    if dtype =='datetime64[ns]':
        index = pd.to_datetime(ds.time.values).year
    elif dtype == 'float64':
        index = ds.time.values.astype(int)
    
    df = pd.DataFrame(index=index, columns=['Reference temperature', 'Arctic temperature'])
    
    df['Reference temperature'] = temp_ref
    df['Arctic temperature'] = temp_arctic
    

    return df