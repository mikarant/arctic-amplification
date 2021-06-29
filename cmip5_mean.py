#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:35:38 2020

@author: rantanem
This script calculates CMIP5 multi-model mean temperature field and outputs it
as a netcdf file
"""
import xarray as xr
import pandas as pd

# open dataset                  
cmip5 = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/cmip5/rcp85.nc')


models = []

timeaxis = pd.date_range(start='1850-12-31', end='2101-01-01', freq='Y')


# loop over all models
for m in cmip5['source_id']:
    print('Calculating ' + str(m.values))
    
    # select temperature
    temp = cmip5['tas'].sel(source_id=m).squeeze().drop('member_id')

    ds_name = '/home/rantanem/Documents/python/data/arctic_warming/'+str(temp.source_id.values)+'_cmip5.nc'
    ds_temp = temp.to_dataset(name='tas')
    ds_temp = ds_temp.rename({'year':'time'}).assign_coords({"time":timeaxis })
    ds_temp.lat.attrs={'standard_name':"latitude",'units':'degrees_north','long_name':'Latitude'}
    ds_temp.lon.attrs={'standard_name':"longitude",'units':'degrees_east','long_name':'Longitude'}

    ds_temp.to_netcdf(path=ds_name, mode='w', format='NETCDF4')
    models.append(temp)
    
model_mean = xr.concat(models, dim='source_id').mean(dim='source_id').squeeze()


ds = model_mean.to_dataset(name='tas')

timeaxis = pd.date_range(start='1850-01-01', end='2101-01-01', freq='1Y')
ds = ds.rename({'year':'time'}).assign_coords({"time":timeaxis })

## add attributes for the coordinates
ds.lat.attrs={'standard_name':"latitude",'units':'degrees_north','long_name':'Latitude'}
ds.lon.attrs={'standard_name':"longitude",'units':'degrees_east','long_name':'Longitude'}


ds.to_netcdf(path='/home/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_mean.nc', mode='w', format='NETCDF4')