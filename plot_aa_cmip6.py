#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:32:42 2020

@author: rantanem
"""
from os import listdir
from os.path import isfile, join
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import functions as fcts
from cdo import *
cdo = Cdo()
from scipy import stats
import seaborn
seaborn.reset_orig()


### lenght of period (default = 40 years)
period = 40

# starting year (the first year which is included for the linear trends)
startYear = 1961

# latitude threshold for the Arctic
latitude_threshold = 66.5

# reference area ('global', 'nh' or 'sh')
refarea = 'global'


model_stats = pd.read_excel('/home/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_stats_selected.xlsx',
                            engine='openpyxl')

## choose cmip5 or cmip6 models
modelGeneration = 'cmip6'

##### sort by ECS
## upper models
q_upper = 0.66
upper =  model_stats['ECS'] > model_stats['ECS'].quantile(q_upper)
high_cat = list(model_stats['MODEL'][upper])

## lower models
q_lower = 0.33
lower =  model_stats['ECS'] < model_stats['ECS'].quantile(q_lower)
low_cat = list(model_stats['MODEL'][lower])

## middle models
middle =  (model_stats['ECS'] > model_stats['ECS'].quantile(q_lower)) \
           & (model_stats['ECS'] < model_stats['ECS'].quantile(q_upper))
mid_cat = list(model_stats['MODEL'][middle])


# choose the scenario
ssp='ssp245'

if modelGeneration == 'cmip5':
    ssp = 'rcp85'



# variable
var = 'tas'

# open model dataset
if modelGeneration == 'cmip6':
    import re
    ssp_path = '/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+ssp+'/'
    files = [f for f in listdir(ssp_path) if isfile(join(ssp_path, f))]
    full_files = [ssp_path + f for f in files]
    models = [re.sub('\.nc','',f) for f in files]

    # open one model dataset
    ds =  xr.open_dataset(full_files[0])
elif modelGeneration == 'cmip5':
    cmip6 = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/cmip5/'+ssp+'.nc')
else:
    print('Choose either cmip5 or cmip6')
    sys.exit()
    


# select observational datasets
obsDatasets = ['BEST', 'GISTEMP', 'COWTAN','ERA5']

# variables in observations
variables = {'GISTEMP': 'tempanomaly',
             'BEST': 'temperature',
             'COWTAN': 'temperature_anomaly',
             'ERA5': 't2m',
             }


if refarea =='global':
    cond = ds.lat>=-90
elif refarea =='nh':
    cond = ds.lat>=0
elif refarea =='sh':
    cond = ds.lat<=0
    
# weights for the model grid
weights = np.cos(np.deg2rad(ds.lat))



years = np.arange(startYear,2100-period+1)

mod = models

df =pd.DataFrame(index=years+period-1, columns=mod)
df_slope =pd.DataFrame(index=years+period-1, columns=mod)
df_slope_a =pd.DataFrame(index=years+period-1, columns=mod)

cmip_years = np.unique(pd.to_datetime(ds.time.values).year)
temp_arctic = pd.DataFrame(index=cmip_years, columns=mod)
temp = pd.DataFrame(index=cmip_years, columns=mod)


# loop over all models
for m in mod:
    print(m)
    yearmean = cdo.yearmean(input=ssp_path+m+'.nc')
    ds = xr.open_dataset(yearmean)
    

    
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
        
## print global mean and arctic mean trends ffrom  models
print('Global mean trend 1980-2019:')
print(str(np.round(df_slope.loc[2019].mean(),4)))
print('Arctic mean trend 1980-2019:')
print(str(np.round(df_slope_a.loc[2019].mean(),4)))
print('Arctic amplification 1980-2019:')
print(str(np.round(df.loc[2019].mean(),4)))

### observations

## initialize dataframes
temp_obs_arctic = pd.DataFrame(index=np.arange(startYear,2020), columns=obsDatasets)
temp_obs_ref = pd.DataFrame(index=np.arange(startYear,2020), columns=obsDatasets)

# loop over datasets
for o in obsDatasets:
    # get temperatures in the reference area and arctic and put them to the dataframes
    df_temp = fcts.getObsTemps(o, variables[o], latitude_threshold, refarea)
    
    temp_obs_arctic[o] = df_temp.loc[startYear:,'Arctic temperature']
    temp_obs_ref[o] = df_temp.loc[startYear:,'Reference temperature']


# calculate AA ratios for the observational datasets
years = np.arange(startYear,2020-period+1)
df_obs =pd.DataFrame(index=years+period-1, columns=obsDatasets)

for y in years:
    yrange = np.arange(y,y+period)
    
    for o in obsDatasets:
        r = fcts.getRatioObs(temp_obs_ref, temp_obs_arctic, o, yrange, period)
        df_obs[o][y+period-1] = r
        
    
### export trends in csv-file
yrange =  np.arange(1980,2020)
df_reference_temps = pd.DataFrame(index=yrange, columns=obsDatasets)
df_arctic_temps = pd.DataFrame(index=yrange, columns=obsDatasets)

df_trends = pd.DataFrame(index=['Arctic trend', 'Global trend'], columns=obsDatasets)

for o in obsDatasets:
    f = temp_obs_ref[o][yrange]
    f_a = temp_obs_arctic[o][yrange]
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f)
    slope_a, _, _, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    
    df_arctic_temps[o] = f_a
    df_reference_temps[o] = f

    df_trends[o]['Arctic trend'] = slope_a
    df_trends[o]['Global trend'] = slope

df_trends.to_csv('/home/rantanem/Documents/python/data/arctic_warming/observed_trends.csv')
df_arctic_temps.to_csv('/home/rantanem/Documents/python/data/arctic_warming/arctic_temps_obs.csv', index_label='Year')
df_reference_temps.to_csv('/home/rantanem/Documents/python/data/arctic_warming/reference_temps_obs.csv', index_label='Year')

###################################
# PLOT RESULTS
###################################
   
 
upper_bondary = np.array(np.mean(df, axis=1) + np.std(df, axis=1)*1.6448, dtype=float) 
lower_bondary = np.array(np.mean(df, axis=1) - np.std(df, axis=1)*1.6448, dtype=float) 

    
plt.figure(figsize=(9,6), dpi=200)
ax=plt.gca()
plt.plot(df.loc[:,:] ,linewidth=1)
# plt.plot(df.loc[:,'CanESM5'],linewidth=1, label='Single models')

plt.plot(df.mean(axis=1),'k' ,linewidth=2, label='CMIP6 mean')
# plt.plot(df.max(axis=1),'k' ,linewidth=0.5, label='CMIP6 max/min')
# plt.plot(df.min(axis=1),'k' ,linewidth=0.5)


ax.fill_between(df.index,  upper_bondary,lower_bondary, 
                where=upper_bondary >=  lower_bondary,
                facecolor='grey', interpolate=True,
                zorder=4,alpha=0.4)

plt.plot(df_obs.mean(axis=1), '-',linewidth=3, color='r', zorder=5,label='Obs')
plt.axvline(x=2014, color='k',linestyle='--')
# set xticks every ten year
plt.xticks(np.arange(2000,2100,10), fontsize=14)
plt.yticks(np.arange(-1,7,1), fontsize=14)
plt.ylim(-1,6)
plt.xlim(2000,2099)
plt.grid(False,zorder=10)
plt.ylabel('Arctic amplification', fontsize=14)
if mod == high_cat:
    # plt.title('High sensitivity models in '+ssp)
    figureName = 'cmip_aa_'+ssp+'_high_cat.png'
elif mod == mid_cat:
    # plt.title('Middle sensitivity models in '+ssp)
    figureName = 'cmip_aa_'+ssp+'_mid_cat.png'
elif mod == low_cat:
    # plt.title('Low sensitivity models in '+ssp)
    figureName = 'cmip_aa_'+ssp+'_low_cat.png'
else:
    # plt.title('All models in '+ssp)
    figureName = 'cmip_aa_'+ssp+'_all.png'

plt.scatter(2019, df_obs.mean(axis=1)[2019], s=40, c='r', zorder=5)
# plt.annotate('AA in 2019: 3.8', (2021, 3.8), xycoords='data', va='center', color='r')


plt.legend(ncol=3, loc='upper center',fontsize=14,  bbox_to_anchor=(0.5, 1.1), frameon=False)

figurePath = '/home/rantanem/Documents/python/figures/'
   
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')

# models_to_csv = [ 'FGOALS-f3-L','FIO-ESM-2-0','MRI-ESM2-0','CAMS-CSM1-0','GFDL-ESM4','GFDL-CM4',
#                  'CanESM5','CESM2-WACCM','IPSL-CM6A-LR','BCC-CSM2-MR','MIROC6',
#                  'INM-CM5-0','NorESM2-LM']
  
# df.loc[:,models_to_csv].to_csv('/home/rantanem/Downloads/aa_'+ssp+'.csv')
# cond = df >= 3.8
# cs = (cond.sum(axis=1))# / np.shape(df)[1]) * 100
# cs.plot()



obs_mean_a = temp_obs_arctic.mean(axis=1)
clim_obs_a = obs_mean_a.loc[1981:2010].mean()
obs_anom_a = obs_mean_a-clim_obs_a
obs_mean = temp_obs_ref.mean(axis=1)
clim_obs = obs_mean.loc[1981:2010].mean()
obs_anom = obs_mean-clim_obs


 

cmap = plt.get_cmap("tab10")

plt.figure(figsize=(9,5), dpi=200)
ax=plt.gca()
plt.plot(temp.mean(axis=1)[np.arange(startYear,2099)], label='Global mean CMIP6')


ax.fill_between(np.arange(startYear,2099),  
                (temp.mean(axis=1) + 1.6448*temp.std(axis=1))[np.arange(startYear,2099)],
                (temp.mean(axis=1) - 1.6448*temp.std(axis=1))[np.arange(startYear,2099)], 
                where=(temp.mean(axis=1) + 1.6448*temp.std(axis=1))[np.arange(startYear,2099)]
                >= (temp.mean(axis=1) - 1.6448*temp.std(axis=1))[np.arange(startYear,2099)],
                facecolor=cmap(0), interpolate=True,
                zorder=0,alpha=0.4)
# plt.plot(temp['CAMS-CSM1-0'][np.arange(startYear,2099)], label='Global mean CAMS-CSM1-0',
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
# plt.plot(temp_arctic['CAMS-CSM1-0'][np.arange(startYear,2099)], label='Arctic mean CAMS-CSM1-0',
#           color=cmap(5))

plt.plot(obs_anom,label='Global mean obs', color=cmap(2))
# plt.plot(obs_anom_a, label='Arctic mean obs', color=cmap(3))



# plt.plot(df.mean(axis=1),'k' ,linewidth=2, label='AA in ')
plt.ylim(-1,1.5)
plt.xticks(np.arange(1900,2110,10))
plt.xlim(1965,2025)
plt.grid(True)
plt.ylabel('Temperature anomaly (1981-2010) [°C]', fontsize=14)

plt.legend(ncol=2, loc='upper left',)

if mod == high_cat:
    plt.title('High sensitivity models in '+ssp)
    figureName = 'cmip_temp_'+ssp+'_high_cat.png'
elif mod == mid_cat:
    plt.title('Middle sensitivity models in '+ssp)
    figureName = 'cmip_temp_'+ssp+'_mid_cat.png'
elif mod == low_cat:
    plt.title('Low sensitivity models in '+ssp)
    figureName = 'cmip_temp_'+ssp+'_low_cat.png'
else:
    # plt.title('All models in '+ssp)
    figureName = 'cmip_temp_'+ssp+'_all.png'
   
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


diff = df.loc[2019] -df_obs.loc[2019].mean()
diff.sort_values()


a =  pd.DataFrame(df_obs.mean(axis=1)).values
b = df.loc[2000:2019].astype(float)







model_ecs = model_stats[['MODEL','ECS']]
model_ecs.index= model_ecs['MODEL']
model_ecs= model_ecs.drop(columns=['MODEL'])

b = pd.DataFrame(df.loc[2000:2019].max(), columns=['aa'], index=df.loc[2019].index)
maxobs = pd.DataFrame(df_obs.mean(axis=1).max(), columns=['aa'], index=['Observed'])
maxmean = pd.DataFrame(df.loc[2000:2019].max().mean(), columns=['aa'], index=['CMIP5 mean'])


ecs_aa = model_ecs.join(b)


  




aavalues = maxobs.append(maxmean).append(b)


plt.figure(figsize=(9,5), dpi=200)
ax=plt.gca()
barlist = plt.bar(x=aavalues.index, height=aavalues.values.squeeze(), facecolor='lightblue')
barlist[0].set_color('r')
barlist[1].set_color('b')
plt.xticks( rotation='vertical')

plt.ylabel('Maximum Arctic amplification ratio by 2019\ncalculated with 40-year linear trends')
figureName = 'max_aa.png'
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')


aa_t = pd.DataFrame(index=df.columns, columns=['aa','T'])

for m in df.columns:
    aa = df[m].loc[2000:2019]
    T = temp[m].loc[2000:2019]
    d_aa, _, _, p_value, stderr = stats.linregress(aa.index, aa.astype(float).values)
    d_T, _, _, p_value, stderr = stats.linregress(T.index, T.astype(float).values)
    # max_aa_year = df[m].astype(float).idxmax()
    # temp_anom = temp[m].loc[max_aa_year]
    aa_t.loc[m].aa = d_aa
    aa_t.loc[m].T = d_T

aa = df_obs.mean(axis=1).loc[2000:2019]
T = obs_anom.loc[2000:2019]
d_aa, _, _, p_value, stderr = stats.linregress(aa.index, aa.astype(float).values)
d_T, _, _, p_value, stderr = stats.linregress(T.index, T.astype(float).values)      
  
aa_t = aa_t.append(pd.Series({'aa': d_aa, 'T': d_T}, name='Observed'))

aa_t.to_csv('/home/rantanem/Documents/python/data/arctic_warming/'+ ssp+'_dAAdT.csv')


