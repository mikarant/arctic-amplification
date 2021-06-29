#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:25:17 2020

@author: rantanem

This script calculates the multi-model mean annual temperature for CMIP6 models

"""
import xarray as xr
import pandas as pd
from cdo import *
cdo = Cdo()



scenario = 'ssp245'

# open model names
model_stats = pd.read_excel('/home/rantanem/Documents/python/data/arctic_warming/cmip6/ssp245/MODEL_NAMES.xlsx', engine='openpyxl')


models=[]

y1=1980
y2=2019
timeaxis = pd.date_range(start=str(y1)+'-12-31', end=str(y2+1)+'-01-01', freq='Y')



# loop over all models
for m in model_stats.values.squeeze():
    
    print('Calculating ' + m)
    
    filename= '/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+scenario+'/'+str(m)+'.nc'

    annual_means = cdo.yearmean(input =  '-selyear,'+str(y1)+'/'+str(y2)+' ' + filename)
        
    ds =  xr.open_dataset(annual_means)
        # select temperature
    temp = ds['tas'].squeeze().drop(['height'], errors='ignore').assign_coords({'time':timeaxis})
    temp['model'] = m


    models.append(temp)
    
    
model_da = xr.concat(models, dim='model')
    
model_mean = model_da.mean(dim='model').squeeze()





ds = model_mean.to_dataset(name='tas')

ds.lat.attrs={'standard_name':"latitude",'units':'degrees_north','long_name':'Latitude'}
ds.lon.attrs={'standard_name':"longitude",'units':'degrees_east','long_name':'Longitude'}

# output the model-mean field as netcdf
ds.to_netcdf(path='/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+scenario+'/cmip6_'+ scenario+'_mean.nc', mode='w', format='NETCDF4')