#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:32:42 2020

@author: rantanem
## this is the main script which calculates the AA diagnostics for the observations and the models.
## written by Mika Rantanen

"""
# import relevat modules
import os
from os import listdir
from os.path import isfile, join
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fcts
import plots
from cdo import *
cdo = Cdo()
from scipy import stats
import seaborn
seaborn.reset_orig()


### lenght of trend period (default = 40 years)
period = 40

# starting year (the first year which is included for the linear trends)
startYear = 1961

# latitude threshold for the Arctic
latitude_threshold = 66.5

# reference area ('global', 'nh' or 'sh')
refarea = 'global'

# Month/season (2,' DJF' etc. For annual, select 'ANN')
season = 'ANN'


## choose cmip5 or cmip6 models
modelGeneration = 'cmip6'

# choose the scenario
ssp='ssp245'


# variable
var = 'tas'

# open model dataset
if modelGeneration == 'cmip6':
    # import re
    ssp_path = '/Users/rantanem/Documents/python/data/arctic_warming/cmip6/'+ssp+'/'
    files = [f for f in listdir(ssp_path) if isfile(join(ssp_path, f))]
    full_files = [ssp_path + f for f in files]
    # models = [re.sub('\.nc','',f) for f in files]
    models = pd.read_excel('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/ssp245/MODEL_NAMES.xlsx',
                           engine='openpyxl').values.squeeze()

    
    sic_path = ssp_path+'sic_kimmo/'
    # sic_files = [f for f in listdir(sic_path) if isfile(join(sic_path, f))]
    # sic_full_files = [sic_path + f for f in sic_files]
    # sic_models = [re.sub('\.nc','',f) for f in sic_files]


    # open one model dataset
    ds =  xr.open_dataset(full_files[1])
else:
    print('Choose cmip6')
    sys.exit()
    


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



years = np.arange(startYear,2100-period+1)

mod = models

## AA based on linear trend ratio
df =pd.DataFrame(index=years+period-1, columns=mod)
df_slope =pd.DataFrame(index=years+period-1, columns=mod)
df_slope_a =pd.DataFrame(index=years+period-1, columns=mod)
df_slope_sic =pd.DataFrame(index=years+period-1, columns=mod)

## AA based on difference in T anomalies
df_anom =pd.DataFrame(index=years, columns=mod)

cmip_years = np.unique(pd.to_datetime(ds.time.values).year)
temp_arctic = pd.DataFrame(index=cmip_years, columns=mod)
temp = pd.DataFrame(index=cmip_years, columns=mod)
siconc = pd.DataFrame(index=cmip_years, columns=mod)
mean_sic = pd.DataFrame(index=np.arange(1850, 2100), columns=mod)


# loop over all models
for m in mod[:]:
    print('Calculating ' + m +' model')
    yearmean = cdo.yearmean(input=operator+season+' '+ ssp_path+m+'.nc')
    rm = cdo.remapcon('r720x360 -selvar,tas ',input=yearmean, options='-b F32')
    ds = xr.open_dataset(rm)
    
    # weights for the model grid
    weights = np.cos(np.deg2rad(ds.lat))
    
    if refarea =='global':
        cond = ds.lat>=-90
    elif refarea =='nh':
        cond = ds.lat>=0
    elif refarea =='sh':
        cond = ds.lat<=0
    
    
    ### check if sea ice is found
    sic_model = m + '.nc'
    sic_fullpath = sic_path + sic_model
    sic_exists = os.path.exists(sic_fullpath)
    if sic_exists:
        print('Sea ice exists\n')
        cdoinput =  '-selyear,1850/2099 -fldsum -sellonlatbox,0,360,0,90 -mul -divc,100 '+sic_fullpath+' -gridarea '+sic_fullpath
        iceareafile = cdo.yearmean(input =operator+season+' '+ cdoinput)
        # iceareafile = cdo.yearmean(input ='-selmon,9 '+ cdoinput)
        ds_sic = xr.open_dataset(iceareafile)

        sic = ds_sic['siconc'].squeeze()
        mean_sic[m] = sic.values
        clim = sic.sel(time=slice("1981-01-01", "2010-12-31")).mean()
        
        siconc[m] = sic-clim
        for y in years:
            yrange = np.arange(y,y+period)
            s = siconc[m][yrange]
            slope_sic, _, _, p_value_sic, _ = stats.linregress(yrange, s.values)
            df_slope_sic[m][y+period-1] = slope_sic
    else:
        print('No sea ice available\n')
    # select temperature
    t = ds[var].where(cond).weighted(weights).mean(("lon", "lat")).squeeze()
    
    clim = t.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    temp[m] = t - clim
    # select temperature
    
    t_a = ds[var].where(ds.lat>=latitude_threshold).weighted(weights).mean(("lon", "lat")).squeeze()
    clim = t_a.sel(time=slice("1981-01-01", "2010-12-31")).mean()
    temp_arctic[m] = t_a - clim
    
    for y in years:
        yrange = np.arange(y,y+period)
        f = temp[m][yrange]
        f_a = temp_arctic[m][yrange]
        slope, _, _, p_value, _ = stats.linregress(yrange, f.values)
        slope_a, _, _, p_value_a, _ = stats.linregress(yrange, f_a.values)
        ratio = slope_a/slope
        df[m][y+period-1] = ratio
        df_slope[m][y+period-1] = slope
        df_slope_a[m][y+period-1] = slope_a
       
        df_anom[m][y] = temp_arctic[m][y] - temp[m][y]
    
    os.remove(yearmean)
    os.remove(rm)
    

## calculate observed sea ice trend using ERA5
# file = '/home/rantanem/Downloads/ecv_seaice/HadISST.2.2.0.0_sea_ice_concentration.nc'
file = '/Users/rantanem/Downloads/ecv_seaice/mean/all.nc'
cdoinput =  ' -fldsum -sellonlatbox,0,360,0,90 -mul -selvar,ci '+file+' -gridarea -selvar,ci '+file
era5_siafile = cdo.yearmean(input =operator+season+' '+ cdoinput)

## calculate observed sea ice trend using UHH product
file = '/Users/rantanem/Downloads/SeaIceArea__NorthernHemisphere__monthly__UHH__v2019_fv0.01.nc'
sia_file = cdo.yearmean(input=operator+season+' '+file)


# define the observational sia datasets
sia_obs_names = ['Bootstrap','NASATeam','OSISAF','HadISST','Walsh']


sia_slopes = pd.DataFrame(index=sia_obs_names, columns=['SIA'])
sia_slopes_max = pd.DataFrame(index=sia_obs_names, columns=['SIA'])
sia_areas = pd.DataFrame(index=sia_obs_names, columns=['SIA'])
sia_values = pd.DataFrame(columns=sia_obs_names, index=np.arange(1980,2020))
sia_trend_errs = pd.DataFrame(index=sia_obs_names, columns=['SIA'])


sia_slopes.loc['Bootstrap'], sia_values['Bootstrap'],sia_trend_errs.loc['Bootstrap']  = fcts.get_sia_trend(sia_file, 'nsidc_bt', period=[1980,2019], factor=1e6*1000*1000)
sia_slopes.loc['NASATeam'], sia_values['NASATeam'],sia_trend_errs.loc['NASATeam']  = fcts.get_sia_trend(sia_file, 'nsidc_nt', period=[1980,2019], factor=1e6*1000*1000)
sia_slopes.loc['OSISAF'], sia_values['OSISAF'],sia_trend_errs.loc['OSISAF']  = fcts.get_sia_trend(sia_file, 'osisaf', period=[1980,2019],factor=1e6*1000*1000)
sia_slopes.loc['Walsh'], sia_values['Walsh'],sia_trend_errs.loc['Walsh']  = fcts.get_sia_trend(sia_file, 'walsh', period=[1980,2019],factor=1e6*1000*1000)
sia_slopes.loc['HadISST'], sia_values['HadISST'],sia_trend_errs.loc['HadISST']  = fcts.get_sia_trend(sia_file, 'HadISST_orig', period=[1980,2019],factor=1e6*1000*1000)

sia_slopes_max.loc['Bootstrap'], _, _  = fcts.get_sia_trend(sia_file, 'nsidc_bt', period=[1979,2018], factor=1e6*1000*1000)
sia_slopes_max.loc['NASATeam'], _, _    = fcts.get_sia_trend(sia_file, 'nsidc_nt', period=[1979,2018],factor=1e6*1000*1000)
sia_slopes_max.loc['OSISAF'], _, _  = fcts.get_sia_trend(sia_file, 'osisaf', period=[1979,2018],factor=1e6*1000*1000)
sia_slopes_max.loc['Walsh'], _, _   = fcts.get_sia_trend(sia_file, 'walsh', period=[1979,2018],factor=1e6*1000*1000)
sia_slopes_max.loc['HadISST'], _, _   = fcts.get_sia_trend(sia_file, 'HadISST_orig', period=[1979,2018],factor=1e6*1000*1000)


sia_areas.loc['Bootstrap'] = fcts.get_sia_area(sia_file, 'nsidc_bt', factor=1e6*1000*1000)
sia_areas.loc['NASATeam']  = fcts.get_sia_area(sia_file, 'nsidc_nt', factor=1e6*1000*1000)
sia_areas.loc['OSISAF']  = fcts.get_sia_area(sia_file, 'osisaf', factor=1e6*1000*1000)
sia_areas.loc['Walsh']  = fcts.get_sia_area(sia_file, 'walsh', factor=1e6*1000*1000)
sia_areas.loc['HadISST']  = fcts.get_sia_area(sia_file, 'HadISST_orig', factor=1e6*1000*1000)

sia_standard_errors = sia_values.std()/np.sqrt(40)

## print global mean and arctic mean trends from the models
print('Modelled global mean trend 1980-2019:')
print(str(np.round(df_slope.loc[2019].mean()*10,5)) + '°C per decade')
print('Modelled Arctic mean trend 1980-2019:')
print(str(np.round(df_slope_a.loc[2019].mean()*10,5)) + '°C per decade')
print('Modelled Arctic amplification 1980-2019:')
print(str(np.round(df.loc[2019].mean(),5)))

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
## AA using difference in SAT anomalies
df_diff =pd.DataFrame(index=temp_obs_ref.index, columns=obsDatasets)

for y in years:
    yrange = np.arange(y,y+period)
    
    for o in obsDatasets:
        r,int_min,int_max = fcts.getRatioObs(temp_obs_ref, temp_obs_arctic, o, yrange, period)
        df_obs[o][y+period-1] = r
        df_obs_min[o][y+period-1] = int_min
        df_obs_max[o][y+period-1] = int_max

for y in temp_obs_ref.index:
    for o in obsDatasets:
        diff = temp_obs_arctic[o][y] - temp_obs_ref[o][y]
        df_diff[o][y] = diff
        
print('Observed Arctic amplification 1980-2019:')
print(str(np.round(df_obs.loc[2019].mean(),5)))
    
### export trends in csv-file
syear = 2019-period
yrange =  np.arange(syear,2019) 

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


# df_reference_temps = pd.DataFrame(index=yrange, columns=obsDatasets)
# df_arctic_temps = pd.DataFrame(index=yrange, columns=obsDatasets)

# for o in obsDatasets:
#     f = temp_obs_ref[o][yrange]
#     f_a = temp_obs_arctic[o][yrange]

#     df_arctic_temps[o] = f_a
#     df_reference_temps[o] = f

df.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_simulated_aa.csv', index_label='Year')
df_trends.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends.csv')
df_err.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_errors.csv')

temp_obs_arctic.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/arctic_temps_obs.csv', index_label='Year')
temp_obs_ref.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/reference_temps_obs.csv', index_label='Year')

### ### read calculated (by Otto) error bounds for observational results
cmip6_errors = pd.read_csv('https://raw.githubusercontent.com/mikarant/arctic-amplification/main/bootstrapCI_temps_obs_19802019.csv',index_col=0)
    
# replace the calculated values by Otto's bootstrapping work
df_obs_min.loc[2019] = cmip6_errors['CIlowerPercentile']
df_obs_max.loc[2019] = cmip6_errors['CIupperPercentile']
df_obs.loc[2019] = cmip6_errors['ratio']

df_obs_min.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_min.csv', index_label='Year')
df_obs_max.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_max.csv', index_label='Year')
df_obs.to_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa.csv', index_label='Year')



###################################
# PLOT RESULTS
###################################


plots.plot_trends(df_trends, df_err, df_slope_a, df_slope, season, annot=True)

plots.plot_trends_aa(df_trends, df_err, df_slope_a, df_slope, df_obs, df_obs_min, df_obs_max, df, season, annot=False)

df_slope_new = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_global_trends.csv', index_col=0)
df_slope_a_new = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_arctic_trends.csv', index_col=0)


plots.trend_plot_2d(df_trends, df_err, df_slope_a_new, df_slope_new, 'CMIP6 models')


plots.sia_tas_scatter(df_slope_sic, df_slope_a, df_trends, sia_slopes, sia_trend_errs, df_err)

plots.sia_aa_scatter(df_slope_sic, df, df_obs, sia_slopes_max, df_slope, df_trends, sia_trend_errs)

plots.meansia_aa_scatter(df, df_slope_sic, df_obs, mean_sic, sia_areas, sia_standard_errors)

plots.plot_fig_4(df_trends, df_err, df_slope_a, df_slope, df_obs, df_obs_min, df_obs_max, df, annot=False)


df_new = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)


plots.plot_pdf(df, df_obs, 2019)

plots.plot_pdf_ens(df_obs, 2018)

plots.plot_pdf_ens_time_range(df_obs, 2018, 2000, 2040)

plots.plot_time_series_cmip(df, df_obs)

plots.plot_time_series_ens(df_obs)



miroc_arctic=pd.DataFrame(data=canesm_ds.arctic_trend[:, -100:].transpose().values.squeeze()/10, 
                               index=np.arange(2000, 2100))

miroc_global=pd.DataFrame(data=canesm_ds.global_trend[:, -100:].transpose().values.squeeze()/10, 
                               index=np.arange(2000, 2100))


plots.trend_plot_2d(df_trends, df_err, miroc_arctic, miroc_global, 'CanESM5 ensemble')

    
# plt.figure(figsize=(9,6), dpi=200)
# ax=plt.gca()
# plt.plot(df.loc[:,:] ,linewidth=1)
# plt.plot(df.loc[:,'INM-CM5-0'],linewidth=2, label='Single models')


# plt.plot(df.mean(axis=1),'k' ,linewidth=2, label='CMIP6 mean')
# # plt.plot(df.max(axis=1),'k' ,linewidth=0.5, label='CMIP6 max/min')
# # plt.plot(df.min(axis=1),'k' ,linewidth=0.5)


# ax.fill_between(df.index,  upper_bondary,lower_bondary, 
#                 where=upper_bondary >=  lower_bondary,
#                 facecolor='grey', interpolate=True,
#                 zorder=4,alpha=0.4)

# plt.plot(df_obs.mean(axis=1), '-',linewidth=3, color='r', zorder=5,label='Obs')
# plt.axvline(x=2014, color='k',linestyle='--')
# # set xticks every ten year
# plt.xticks(np.arange(2000,2100,10), fontsize=14)
# plt.yticks(np.arange(-1,7,1), fontsize=14)
# plt.ylim(-1,6)
# plt.xlim(2000,2099)
# plt.grid(False,zorder=10)
# plt.ylabel('Arctic amplification', fontsize=14)

# figureName = 'cmip_aa_'+ssp+'_all.png'

# plt.scatter(2019, df_obs.mean(axis=1)[2019], s=40, c='r', zorder=5)
# # plt.annotate('AA in 2019: 3.8', (2021, 3.8), xycoords='data', va='center', color='r')


# plt.legend(ncol=3, loc='upper center',fontsize=14,  bbox_to_anchor=(0.5, 1.1), frameon=False)

# figurePath = '/home/rantanem/Documents/python/figures/'
   
# plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')

# models_to_csv = [ 'FGOALS-f3-L','MRI-ESM2-0','CAMS-CSM1-0','GFDL-ESM4',
#                   'CanESM5','CESM2-WACCM','IPSL-CM6A-LR','BCC-CSM2-MR','MIROC6',
#                   'INM-CM5-0','NorESM2-LM', 'CMCC-CM2-SR5', 'NorESM2-MM', 'NESM3',
#                   'NorESM2-MM', 'CMCC-CM2-SR5']
  
# df.loc[:,models_to_csv].to_csv('/home/rantanem/Downloads/aa_'+ssp+'.csv')

# df_slope.loc[:,models_to_csv].to_csv('/home/rantanem/Downloads/ref_trends_'+ssp+'.csv')
# # cond = df >= 3.8
# # cs = (cond.sum(axis=1))# / np.shape(df)[1]) * 100
# # cs.plot()



# obs_mean_a = temp_obs_arctic.mean(axis=1)
# clim_obs_a = obs_mean_a.loc[1981:2010].mean()
# obs_anom_a = obs_mean_a-clim_obs_a
# obs_mean = temp_obs_ref.mean(axis=1)
# clim_obs = obs_mean.loc[1981:2010].mean()
# obs_anom = obs_mean-clim_obs


 

# plt.figure(figsize=(9,5), dpi=200)
# ax=plt.gca()
# plt.plot(temp.mean(axis=1)[np.arange(startYear,2099)], label='Global mean CMIP6')


# ax.fill_between(np.arange(startYear,2099),  
#                 (temp.mean(axis=1) + 1.6448*temp.std(axis=1))[np.arange(startYear,2099)],
#                 (temp.mean(axis=1) - 1.6448*temp.std(axis=1))[np.arange(startYear,2099)], 
#                 where=(temp.mean(axis=1) + 1.6448*temp.std(axis=1))[np.arange(startYear,2099)]
#                 >= (temp.mean(axis=1) - 1.6448*temp.std(axis=1))[np.arange(startYear,2099)],
#                 facecolor=cmap(0), interpolate=True,
#                 zorder=0,alpha=0.4)
# plt.plot(temp['INM-CM5-0'][np.arange(startYear,2099)], label='Global mean CAMS-CSM1-0',
#           color=cmap(4))

# plt.plot(temp_arctic.mean(axis=1)[np.arange(startYear,2099)], label='Arctic mean CMIP6',
#           color=cmap(1))

# ax.fill_between(np.arange(startYear,2099),  
#                 temp_arctic.quantile(0.95,axis=1)[np.arange(startYear,2099)],
#                 temp_arctic.quantile(0.05,axis=1)[np.arange(startYear,2099)], 
#                 where=temp_arctic.quantile(0.95,axis=1)[np.arange(startYear,2099)]
#                 >=  temp_arctic.quantile(0.05,axis=1)[np.arange(startYear,2099)],
#                 facecolor=cmap(1), interpolate=True,
#                 zorder=0,alpha=0.4)
# plt.plot(temp_arctic['INM-CM5-0'][np.arange(startYear,2099)], label='Arctic mean CAMS-CSM1-0',
#           color=cmap(5))

# plt.plot(obs_anom,label='Global mean obs', color=cmap(2))
# plt.plot(obs_anom_a, label='Arctic mean obs', color=cmap(3))



# # plt.plot(df.mean(axis=1),'k' ,linewidth=2, label='AA in ')
# # plt.ylim(-1,1.5)
# plt.xticks(np.arange(1900,2110,10))
# plt.xlim(1965,2099)
# plt.grid(True)
# plt.ylabel('Temperature anomaly (1981-2010) [°C]', fontsize=14)

# plt.legend(ncol=2, loc='upper left',)


# figureName = 'cmip_temp_'+ssp+'_all.png'
   
# plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


# diff = df.loc[2019] -df_obs.loc[2019].mean()
# diff.sort_values()


# a =  pd.DataFrame(df_obs.mean(axis=1)).values
# b = df.loc[2000:2019].astype(float)






# syear=2010
# eyear=2040
# range_min = 0.05
# range_max = 0.95

# obsmin = df_obs-df_obs_min
# obsmax = df_obs_max-df_obs

# cmip5_ratios = pd.read_csv('/home/rantanem/Documents/python/data/arctic_warming/cmip5_ratios.csv',index_col=0)

# a = pd.DataFrame(cmip5_ratios.loc[syear:eyear].max(), columns=['aa'], index=cmip5_ratios.loc[2019].index)
# amin = np.mean(a.values) - np.quantile(a,range_min)
# amax =   np.quantile(a,range_max)- np.mean(a.values) 
# aerr = [[amin],[amax]]


# b = pd.DataFrame(df.loc[syear:eyear].max(), columns=['aa'], index=df.loc[2019].index)
# maxobs = pd.DataFrame(df_obs.loc[syear:eyear].max(axis=0).mean(), columns=['aa'], index=['Observed'])
# maxmean = pd.DataFrame(df.loc[syear:eyear].max().mean(), columns=['aa'], index=['CMIP6 mean'])

# bmin = (maxmean - np.quantile(b,range_min)).squeeze()
# bmax =   (np.quantile(b,range_max)- maxmean).squeeze()
# berr = [[bmin],[bmax]]

# aavalues = maxobs.append(maxmean).append(a)

# bbvalues = maxobs.append(maxmean).append(b)

  






# plt.figure(figsize=(7,5), dpi=200)
# ax=plt.gca()

# plt.plot(mean_sic.loc[1950:2099]/(1000*1000*1000000),'grey')

# plt.plot(mean_sic.loc[1950:2099].mean(axis=1)/(1000*1000*1000000),'k',linewidth=3, label='CMIP6 mean (SSP2-4.5)')

# plt.plot(np.arange(1980,2020), sia_values.mean(axis=1)/(1000*1000*1000000),
#           'r', linewidth=2, label='Observations')
# # plt.plot(np.arange(1979,2020), sic_obs2.sel(time=slice("1979-01-01", "2019-12-31"))/(1000*1000*1000000),
# #           'k', linewidth=2, label='Observations (HadISST2)')
# # plt.plot(np.arange(1978,2021), sia_nasa_annual/(1000*1000*1000000),
# #           'r', linewidth=2, label='Observations (NASA)')
# plt.ylabel('Sea ice area [million km²]')
# plt.title('September mean sea ice area in CMIP6 models (SSP2-4.5 scenario)')
# plt.legend()
# # plt.xlim(2000,2016)