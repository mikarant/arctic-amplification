#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:06:22 2021

@author: mprantan
"""
import numpy as np
import pandas as pd
import xarray as xr
import glob
from cdo import *
cdo = Cdo()


# input path for the merged datafiles
path = '/fmi/projappl/project_2003992/cmip6/xr_merged_ssp245/'

# output path for the regridded datafiles
regrid_path = '/fmi/projappl/project_2003992/cmip6/merged_regridded_ssp245/'

# define the years (1850-2014 for hist, 2015-2099 for scen )
years = np.arange(1850, 2100)

# define which normal period is used for the climatology
climate = np.arange(1981,2011)

# define Arctic 
latitude_threshold = 66.5


# read the model names
models = pd.read_excel('/users/mprantan/python/arctic/list_of_models.xlsx')

# number of models
number_of_models = len(models)

# allocate lists
models_global_list = []
models_arctic_list = []

# regrid the model data to 0.5 resolution. here we define the new latitude and longitude
new_lat = np.arange(-90,90.5,0.5)
new_lon = np.arange(0,360,0.5)

# loop over the models
for i in np.arange(0, number_of_models):
    
    m = models.iloc[i]
    
    # list of realizations of that particular model
    list_of_realizations = glob.glob(path + m.Name.rstrip() + '_*.nc')
    
    # loop over the realizations
    for j in range(0, len(list_of_realizations)):
        
        name = list_of_realizations[j].split('/')[-1][:-3]
        print(name)
        
        ds_path = list_of_realizations[j]
        
        realization_name = ds_path.split('/')[-1]
        
        # open the realization and regrid it to the new grid
        time_years = xr.open_dataset(ds_path).time
        output_regrid = regrid_path+realization_name
        # do the regridding
        cdo.remapcon('r720x360 -selvar,tas ', input=ds_path, 
                     output=output_regrid, 
                     options='-b F32')
        
        ds = xr.open_dataset(output_regrid).assign_coords(time=time_years)
        
        
        # weights for areal average
        weights = np.cos(np.deg2rad(ds.lat))
        
        # calculate global average
        t = ds.tas.weighted(weights).mean(('lon','lat')).sel(time=years)
        
        # calculate global temperature anomaly
        clim = t.sel(time=climate).mean()
        temp = t - clim
        
        # calculate Arctic average and its anomaly
        t_a = ds.tas.where(ds.lat>=latitude_threshold).weighted(weights).mean(('lon','lat')).sel(time=years)
        clim = t_a.sel(time=climate).mean()
        temp_a = t_a - clim
        
        # allocate dataframes for the temperatures
        df_global = pd.DataFrame(index=years, columns=[name])
        df_arctic = pd.DataFrame(index=years, columns=[name])
        
        # store the global and arctic temperatures
        df_global[name] = temp.values
        df_arctic[name] = temp_a.values
        
        # append to the list of all realizations
        models_global_list.append(df_global)
        models_arctic_list.append(df_arctic)
   
# convert the lists to dataframes
models_arctic_df = pd.concat(models_arctic_list, axis=1)
models_global_df = pd.concat(models_global_list, axis=1)

# output to csv files
models_global_df.to_csv('/users/mprantan/python/arctic/data/cmip6_global_temps.csv')
models_arctic_df.to_csv('/users/mprantan/python/arctic/data/cmip6_arctic_temps.csv')
