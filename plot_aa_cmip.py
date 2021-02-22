#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 15:32:42 2020

@author: rantanem
"""
import sys
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cdo import *
cdo = Cdo()
from scipy import stats
import seaborn
seaborn.reset_orig()

def getRatioObs(temp_ref, temp_arctic, obsname, yrange, period):

    
    f = temp_obs[obsname][yrange]
    f_a = temp_obs_arctic[obsname][yrange]
    
    slope, _, _, p_value, stderr = stats.linregress(yrange, f.values)
    slope_a, _, _, p_value_a, stderr_a = stats.linregress(yrange, f_a.values)
    ratio = slope_a/slope

    
    return ratio 


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
    cmip6 = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/cmip6/'+ssp+'/'+ssp+'.nc')
elif modelGeneration == 'cmip5':
    cmip6 = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/cmip5/'+ssp+'.nc')
else:
    print('Choose either cmip5 or cmip6')
    sys.exit()
    


# open observational datasets
best = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/BEST_annual.nc')
gistemp = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/GISTEMP_annual.nc')
cw = xr.open_dataset('/home/rantanem/Documents/python/data/arctic_warming/COWTAN_annual.nc')

# weights for model grid
weights = np.cos(np.deg2rad(cmip6.lat))
weights.name = "weights"

models =  list(cmip6['source_id'].values)

### lenght of period (default = 40 years)
period = 30

# starting year (the first year which is included to the linear trends)
startYear = 1961

years = np.arange(startYear,2100-period+1)

mod = models

df =pd.DataFrame(index=years+period-1, columns=mod)
df_slope =pd.DataFrame(index=years+period-1, columns=mod)
df_slope_a =pd.DataFrame(index=years+period-1, columns=mod)

temp_arctic = pd.DataFrame(index=cmip6.year, columns=mod)
temp = pd.DataFrame(index=cmip6.year, columns=mod)


# loop over all models
for m in mod:
    print(m)
    # select temperature
    # t = cmip6[var].sel(source_id=m).drop('member_id').weighted(weights).mean(("lon", "lat")).squeeze()
    t = cmip6[var].sel(source_id=m).weighted(weights).mean(("lon", "lat")).squeeze()
    
    clim = t.sel(year=np.arange(1981,2011)).mean()
    temp[m] = t - clim
    # select temperature
    # t_a = cmip6[var].sel(source_id=m).drop('member_id').where(cmip6.lat>=66.5).weighted(weights).mean(("lon", "lat")).squeeze()
    t_a = cmip6[var].sel(source_id=m).where(cmip6.lat>=66.5).weighted(weights).mean(("lon", "lat")).squeeze()
    clim = t_a.sel(year=np.arange(1981,2011)).mean()
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

### obs
weights_best = np.cos(np.deg2rad(best.latitude))
weights_gistemp = np.cos(np.deg2rad(gistemp.lat))
weights_cw = np.cos(np.deg2rad(cw.latitude))

t_best = best['temperature'].weighted(weights_best).mean(("longitude", "latitude")).squeeze()
t_best_a = best['temperature'].where(best.latitude>=66.5).weighted(weights_best).mean(("longitude", "latitude")).squeeze()

t_gistemp = gistemp['tempanomaly'].weighted(weights_gistemp).mean(("lon", "lat")).squeeze()
t_gistemp_a = gistemp['tempanomaly'].where(gistemp.lat>=66.5).weighted(weights_gistemp).mean(("lon", "lat")).squeeze()

t_cw = cw['temperature_anomaly'].weighted(weights_cw).mean(("longitude", "latitude")).squeeze()
t_cw_a = cw['temperature_anomaly'].where(cw.latitude>=66.5).weighted(weights_cw).mean(("longitude", "latitude")).squeeze()

temp_obs_arctic = pd.DataFrame(t_best_a,index=np.arange(1900,2020), columns=['BEST'])
temp_obs = pd.DataFrame(t_best, index=np.arange(1900,2020), columns=['BEST'])

temp_obs_arctic['GISTEMP'] = t_gistemp_a
temp_obs['GISTEMP'] = t_gistemp
temp_obs_arctic['CW'] = t_cw_a
temp_obs['CW'] = t_cw



years = np.arange(startYear,2020-period+1)
df_obs =pd.DataFrame(index=years+period-1, columns=['BEST', 'GISTEMP', 'CW'])

for y in years:
    yrange = np.arange(y,y+period)
    
    r1 = getRatioObs(temp_obs, temp_obs_arctic, 'BEST', yrange, period)
    r2 = getRatioObs(temp_obs, temp_obs_arctic, 'GISTEMP', yrange, period)
    r3 = getRatioObs(temp_obs, temp_obs_arctic, 'CW', yrange, period)
    
    df_obs['BEST'][y+period-1] = r1
    df_obs['GISTEMP'][y+period-1] = r2
    df_obs['CW'][y+period-1] = r3

  
 
    
###################################
# PLOT RESULTS
###################################
   
 
upper_bondary = np.array(np.quantile(df,q=0.95, axis=1), dtype=float) 
lower_bondary =  np.array(np.quantile(df,q=0.05, axis=1), dtype=float) 
  

    
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
obs_mean = temp_obs.mean(axis=1)
clim_obs = obs_mean.loc[1981:2010].mean()
obs_anom = obs_mean-clim_obs



cmap = plt.get_cmap("tab10")

plt.figure(figsize=(9,5), dpi=200)
ax=plt.gca()
plt.plot(temp.mean(axis=1)[np.arange(startYear,2099)], label='Global mean CMIP6')


ax.fill_between(np.arange(startYear,2099),  
                temp.quantile(0.95,axis=1)[np.arange(startYear,2099)],
                temp.quantile(0.05,axis=1)[np.arange(startYear,2099)], 
                where=temp.quantile(0.95,axis=1)[np.arange(startYear,2099)]
                >=  temp.quantile(0.05,axis=1)[np.arange(startYear,2099)],
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


    

plt.figure(figsize=(9,5), dpi=200)
plt.scatter(x=ecs_aa['ECS'],y=ecs_aa['aa'])


for i in range(0,len(ecs_aa.index)):
    plt.annotate(ecs_aa.index[i], (ecs_aa['ECS'][i], ecs_aa['aa'][i]+.06), fontsize=7)
    
plt.axhline(y=df_obs.mean(axis=1).max(), color='r', linestyle='--')
plt.annotate('Observed', (1.8,3.85), color='r')

plt.axhline(y=ecs_aa['aa'].mean(), color='k', linestyle='--')
plt.annotate('CMIP5 mean', (1.8,ecs_aa['aa'].mean() - 0.15), color='k')
    
plt.xlabel('Equilibrium climate sensitivity [°C]')
plt.ylabel('Maximum Arctic amplification ratio by 2099\ncalculated with 40-year linear trends')
# plt.title('Arctic amplification vs. ECS')

ecs_aa.corr()


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


