#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:25:42 2020

@author: rantanem
This script calculates AA metric for CMIP5 models and outputs the mean AA to netcdf file
"""
import xarray as xr
from cdo import *
cdo = Cdo()

# read model names
cmip5 = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/cmip5/rcp85.nc')

mod = cmip5['source_id'].values

# years = np.arange(1979, 2020, 1)

trendfiles = []


# loop over all models
for m in mod:
    
    print('Calculating '+ m + ' model')

    
    filename= '/home/rantanem/Documents/python/data/arctic_warming/cmip5/'+str(m)+'_cmip5.nc'
    
     # calculate annual means
    annual_means = cdo.yearmean(input =  "-selyear,1980/2019 " + filename)
    
    # calculate trends
    trend1, trend2 = cdo.trend(input = annual_means)
        

    # take the global mean
    output = cdo.fldmean(input = ' '+trend2)
    
     # open the dataset and pick up the global average trend
    ds = xr.open_dataset(output)
    global_mean = ds['tas'].values.squeeze()
    
    
    # divide the global trend by its average
    warming_ratio = cdo.divc(global_mean, input = trend2)
    
    ds_f = xr.open_dataset(warming_ratio)
    aa = ds_f['tas'].squeeze()
    trendfiles.append(aa)
    
# calculate mean AA across the models
aa_mean = xr.concat(trendfiles, dim='source_id').mean(dim='source_id').squeeze()

ds_name = '/home/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_mean_aa.nc'
ds_aa = aa_mean.to_dataset(name='aa')
ds_aa.lat.attrs={'standard_name':"latitude",'units':'degrees_north','long_name':'Latitude'}
ds_aa.lon.attrs={'standard_name':"longitude",'units':'degrees_east','long_name':'Longitude'}

ds_aa.to_netcdf(path=ds_name, mode='w', format='NETCDF4')
