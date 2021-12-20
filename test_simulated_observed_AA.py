#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:41:43 2021

@author: rantanem
"""
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

model='MPI-ESM'
year = 2018

plot_years=np.arange(year, year+1)
    
if model=='MPI-ESM':
    years = np.arange(1889,2100)
else:
    years = np.arange(1889,2101)
ind = np.isin(years, plot_years)

if model=='CMIP6':
    mpi_ds = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)
    canesm_col = [s for s in mpi_ds.columns if 'CanESM5'.rstrip() in s]
    cmip6_without_canesm = [m for m in mpi_ds.columns if m not in canesm_col]
    mpi_ds = mpi_ds[cmip6_without_canesm]
    mpi_ratios_plot=mpi_ds.loc[plot_years].values.transpose()
    aa_ratios = mpi_ds.loc[plot_years]
    all_realizations = aa_ratios.columns
    models = pd.read_excel('/Users/rantanem/Downloads/alexey_models.xlsx')
    models = models[models.ssp245>0]
    models = models[models.Name!='CanESM5']
elif model=='CMIP5':
    mpi_ds = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_ratios.csv',index_col=0)
    mpi_ratios_plot=mpi_ds.loc[plot_years].values.transpose().squeeze()
elif model=='MPI-ESM':
    mpi_ds = xr.open_dataset('/Users/rantanem/Documents/python/data/arctic_warming/data_pdf_plots_'+model+'_rcp45.nc')
    mpi_ratios_plot=mpi_ds.aa.values[:,ind]
elif model=='CanESM5':
    cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)
    canesm_col = [s for s in cmip6.columns if 'CanESM5'.rstrip() in s]
    mpi_ratios_plot = cmip6[canesm_col].loc[year].values.squeeze().ravel()
elif model=='MIROC-ES2L':
    cmip6 = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)
    canesm_col = [s for s in cmip6.columns if 'MIROC-ES2L'.rstrip() in s]
    mpi_ratios_plot = cmip6[canesm_col].loc[year].values.squeeze().ravel()
else:
    print('Model not valid!')



## observed AA; 3.87 for 2019, 4.01 for 2018
AA = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa.csv', index_col=0)
a = AA.loc[year].mean()


distribution_list = []

n_times = 100000
if model=='CMIP6':
    models_with_multiple_real = models[models.ssp245>1]
    for i in np.arange(0,n_times):
    
        # sample with replacement of N runs
        N = random.randrange(2, np.shape(models)[0])
        sample = random.choices(list(models.Name.values), k=N)
        oneruns = []
        for m in sample:
            matching = [s for s in all_realizations if m.rstrip() in s]
            n = random.randrange(0,len(matching))
            onerun = aa_ratios[matching[n]]
            oneruns.append(onerun.values)
        b = np.mean(oneruns)
            
    
        # single run
        
        single_model_i = random.randrange(0, np.shape(models_with_multiple_real)[0])
        single_model = models_with_multiple_real.iloc[single_model_i]
        matching = [s for s in all_realizations if single_model.Name.rstrip() in s]
        n = len(matching)
        
        random_run_i = random.randrange(0, len(matching))
        single_run_name = matching[random_run_i]
        single_run = aa_ratios[single_run_name].values
        
        ensemble_mean = aa_ratios[matching].mean(axis=1).values
        
        c = single_run - ensemble_mean
        
   
        multiplication_factor = (n/(n-1))**0.5
    
        c = c*multiplication_factor
    
        distribution_list.append(a-b+c)
else:
    for i in np.arange(0,n_times):
    
        # sample of N runs with replacement
        N = random.randrange(2, np.shape(mpi_ratios_plot)[0])
        indices = random.choices(np.arange(0,len(mpi_ratios_plot)), k=N)
        sample = mpi_ratios_plot[indices]

        b = np.mean(sample)
    
        # single run
        jth_member = random.randrange(0, np.shape(mpi_ratios_plot)[0])
        single_run = np.squeeze(mpi_ratios_plot[jth_member])
        c = b - single_run
    
        multiplication_factor = (N/(N-1))**0.5
    
        c = c*multiplication_factor
    
        distribution_list.append(a-b+c)  

distribution = np.array(distribution_list)
    
fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=200, sharex=False)
plt.hist(distribution, bins=np.arange(-3,3,0.1), facecolor='darkgrey', edgecolor='k')


cond = np.sum(distribution < 0)
print('P-value: '+str(np.sum(distribution < 0)/n_times))