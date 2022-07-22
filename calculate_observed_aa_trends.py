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
import datetime

# reference area ('global', 'nh' or 'sh')
refarea = 'global'

# Month/season (2,' DJF' etc. For annual, select 'ANN')
season = 'ANN'


### lenght of trend period (default = 43 years)
period = 43

# starting year (the first year which is included for the linear trends)
startYear = 1950

# latitude threshold for the Arctic
latitude_threshold = 66.5

# select observational datasets
obsDatasets = ['BEST', 'GISTEMP', 'HADCRUT','ERA5']

# variables in observations
variables = {'GISTEMP': 'tempanomaly',
             'BEST': 'temperature',
             'HADCRUT': 'tas_mean',
             'ERA5': 't2m',
             }

# long names of observations
longnames = ['Berkeley\nEarth',
             'Gistemp',
             'HadCRUT5',
             'ERA5',
             ]

# derive the month name 
if type(season) == int:
    operator = '-selmon,'
    season = str(season)
    month_name=datetime.date(1900, int(season), 1).strftime('%b').lower()
elif type(season) == str:
    operator = '-selseason,'
    month_name=season.lower()


### observations

## initialize dataframes
temp_obs_arctic = pd.DataFrame(index=np.arange(1950,2022), columns=obsDatasets)
temp_obs_ref = pd.DataFrame(index=np.arange(1950,2022), columns=obsDatasets)

# loop over datasets
for o in obsDatasets:
    # get temperatures in the reference area and arctic and put them to the dataframes
    df_temp = fcts.getObsTemps(o, variables[o], latitude_threshold, refarea, operator, season)
    
    temp_obs_arctic[o] = df_temp.loc[1950:,'Arctic temperature']
    temp_obs_ref[o] = df_temp.loc[1950:,'Reference temperature']


# calculate AA ratios for the observational datasets
years = np.arange(startYear,2022-period+1)
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


# calculate AA ratios for time windows ending to present
years = np.arange(startYear,2021)
AA_to_present = pd.DataFrame(index=years, columns=obsDatasets)
r_to_present = pd.DataFrame(index=years, columns=obsDatasets)


for y in years:
    yrange = np.arange(y,years[-1]+2)
    
    for o in obsDatasets:
        ratio, rvalue = fcts.getRatioPresent(temp_obs_ref, temp_obs_arctic, o, yrange, period)
        AA_to_present[o][y] = ratio
        r_to_present[o][y] = rvalue



        
print('Observed Arctic amplification 1979-2021:')
print(str(np.round(df_obs.loc[2021].mean(),4)))
    
### export 1979-2021 trends in csv-file
syear = 2022-period
yrange =  np.arange(syear,2022) 

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



df_trends.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends_1979-2021_'+month_name+'.csv')
df_err.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_errors_1979-2021_'+month_name+'.csv')
df_obs.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_'+month_name+'.csv', 
              index_label='Year', na_rep='NaN')

AA_to_present.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_to_present.csv', 
                     index_label='Year', na_rep='NaN')

temp_obs_arctic.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/arctic_temps_obs.csv', index_label='Year')
temp_obs_ref.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/global_temps_obs.csv', index_label='Year')