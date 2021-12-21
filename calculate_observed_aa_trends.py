#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:25:29 2021

@author: rantanem
"""
import numpy as np
import pandas as pd
from scipy import stats
import functions as fcts

# reference area ('global', 'nh' or 'sh')
refarea = 'global'

# Month/season (2,' DJF' etc. For annual, select 'ANN')
season = 'ANN'


### lenght of trend period (default = 40 years)
period = 40

# starting year (the first year which is included for the linear trends)
startYear = 1961

# latitude threshold for the Arctic
latitude_threshold = 66.5

# select observational datasets
obsDatasets = ['BEST', 'GISTEMP', 'COWTAN','ERA5']

# variables in observations
variables = {'GISTEMP': 'tempanomaly',
             'BEST': 'temperature',
             'COWTAN': 'temperature_anomaly',
             'ERA5': 't2m',
             }

# long names of observations
longnames = ['Berkeley\nEarth',
             'Gistemp',
             'Cowtan&\nWay',
             'ERA5',
             ]

if type(season) == int:
    operator = '-selmon,'
    season = str(season)
elif type(season) == str:
    operator = '-selseason,'


### observations

## initialize dataframes
temp_obs_arctic = pd.DataFrame(index=np.arange(1950,2020), columns=obsDatasets)
temp_obs_ref = pd.DataFrame(index=np.arange(1950,2020), columns=obsDatasets)

# loop over datasets
for o in obsDatasets:
    # get temperatures in the reference area and arctic and put them to the dataframes
    df_temp = fcts.getObsTemps(o, variables[o], latitude_threshold, refarea, operator, season)
    
    temp_obs_arctic[o] = df_temp.loc[1950:,'Arctic temperature']
    temp_obs_ref[o] = df_temp.loc[1950:,'Reference temperature']


# calculate AA ratios for the observational datasets
years = np.arange(startYear,2020-period+1)
df_obs =pd.DataFrame(index=years+period-1, columns=obsDatasets)
df_obs_min =pd.DataFrame(index=years+period-1, columns=obsDatasets)
df_obs_max =pd.DataFrame(index=years+period-1, columns=obsDatasets)


for y in years:
    yrange = np.arange(y,y+period)
    
    for o in obsDatasets:
        r,int_min,int_max = fcts.getRatioObs(temp_obs_ref, temp_obs_arctic, o, yrange, period)
        df_obs[o][y+period-1] = r
        df_obs_min[o][y+period-1] = int_min
        df_obs_max[o][y+period-1] = int_max


        
print('Observed Arctic amplification 1980-2019:')
print(str(np.round(df_obs.loc[2019].mean(),4)))
    
### export 1980-2019 trends in csv-file
syear = 2020-period
yrange =  np.arange(syear,2020) 

df_trends = pd.DataFrame(index=['Arctic trend', 'Global trend'], columns=obsDatasets)
df_err = pd.DataFrame(index=['Arctic', 'Global'], columns=obsDatasets)

for o in obsDatasets:
    f = temp_obs_ref[o][yrange]
    f_a = temp_obs_arctic[o][yrange]
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f)
    slope_a, _, _, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    
    df_trends[o]['Arctic trend'] = slope_a
    df_trends[o]['Global trend'] = slope
    
    df_err[o]['Arctic'] = stderr_a
    df_err[o]['Global'] = stderr



df_trends.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends.csv')
df_err.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_errors.csv')

temp_obs_arctic.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/arctic_temps_obs.csv', index_label='Year')
temp_obs_ref.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/reference_temps_obs.csv', index_label='Year')