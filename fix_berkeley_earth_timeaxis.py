#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:58:00 2021

@author: rantanem
"""
import sys
import pandas as pd
import xarray as xr
import numpy as np

inputpath = '/Users/rantanem/Documents/python/data/arctic_warming/BEST.nc'
outputpath = '/Users/rantanem/Documents/python/data/arctic_warming/BEST-retimed.nc'

# Open the file as a dataset
ds = xr.open_dataset(inputpath)


# Create a new time axis which better follows the netCDF standards
years = np.arange(1850,2022)

t_axis = pd.date_range(str(years[0])+'-01-01',pd.Timestamp(str(years[-1])+'-12-31'), freq='1M')


# Change the time variable inside the dataset and rewrite to a temp file

ds['time'] = t_axis
ds.to_netcdf(outputpath, format='NETCDF4', encoding={'time': {'dtype': 'i4'}})
