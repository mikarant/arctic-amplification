#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:25:42 2020

@author: rantanem
This script calculates the mean Arctic amplification metric across the CMIP6 models
and outputs the data to netcdf file
"""
import xarray as xr
import pandas as pd
from cdo import *
cdo = Cdo()



scenario = 'ssp245'

model_stats = pd.read_excel('/home/rantanem/Documents/python/data/arctic_warming/cmip6/ssp245/MODEL_NAMES.xlsx', engine='openpyxl')


trendfiles = []
trend = []

y1=1980
y2=2019



# loop over all models
i=1
for m in model_stats.values.squeeze():
    
    print('Calculating '+m+' model')
    
    filename= '/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+scenario+'/'+str(m)+'.nc'
    
     # calculate annual means
    annual_means = cdo.yearmean(input =  '-selyear,'+str(y1)+'/'+str(y2)+' ' + filename)
    
    # calculate trends
    trend1, trend2 = cdo.trend(input = annual_means)
        

    # take the global mean
    output = cdo.fldmean(input = ''+trend2)
    
     # open the dataset and pick up the global average trend
    ds = xr.open_dataset(output)
    global_mean = ds['tas'].values.squeeze()
    
    
    # divide the global trend by its average
    warming_ratio = cdo.divc(global_mean, input = trend2)
    
    ds_f = xr.open_dataset(warming_ratio)
    aa = ds_f['tas'].squeeze().drop(['height'], errors='ignore')
    aa['model'] = m
    trendfiles.append(aa)
    da_trend = xr.open_dataset(trend2)['tas'].squeeze().drop(['height'], errors='ignore')
    
    trend.append(da_trend)
    i+=1
    
# Calculate mean AA across the models    
aa_mean = xr.concat(trendfiles, dim='model').mean(dim='model').squeeze()
aa_std = xr.concat(trendfiles, dim='model').std(dim='model').squeeze()
aa_trend = xr.concat(trend, dim='model').mean(dim='model').squeeze()


ds_name = '/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+scenario+'/'+'cmip6_mean_aa_'+scenario+'.nc'
ds_aa = aa_mean.to_dataset(name='aa')
ds_aa['std'] = aa_std
ds_aa['trend'] = aa_trend*10
ds_aa.lat.attrs={'standard_name':"latitude",'units':'degrees_north','long_name':'Latitude'}
ds_aa.lon.attrs={'standard_name':"longitude",'units':'degrees_east','long_name':'Longitude'}

ds_aa.to_netcdf(path=ds_name, mode='w', format='NETCDF4')
