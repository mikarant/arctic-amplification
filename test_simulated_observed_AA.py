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
import sys

# Select which model dataset you want to test: CMIP5, CMIP6, MPI-GE or CanESM5
model='CMIP5'

# select the ending year of the AA ratio
year = 2021

# Select how many repetitions to use to construct the distributions
n_times = 100000

plot_years=np.arange(year, year+1)
    
if model=='MPI-GE':
    years = np.arange(1889,2100)
else:
    years = np.arange(1889,2101)
ind = np.isin(years, plot_years)

# read modelled AA ratios

if model=='CMIP6':
    df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_ann.csv', index_col=0)
    # exlude CanESM5 model from CMIP6
    canesm_col = [s for s in df.columns if 'CanESM5'.rstrip() in s]
    cmip6_without_canesm = [m for m in df.columns if m not in canesm_col]
    df = df[cmip6_without_canesm]
   
    aa_ratios = df.loc[plot_years]
    all_realizations = df.columns
    models = pd.read_excel('/Users/rantanem/Documents/python/data/arctic_warming/list_of_models.xlsx')
    models = models[models.ssp245>0]
    models = models[models.Name!='CanESM5']
    models_with_multiple_real = models[models.ssp245>1]
elif model=='CMIP5':
    df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa_ann.csv',index_col=0)
    aa_ratios=df.loc[year].values.transpose().squeeze()
elif model=='MPI-GE':
    mpi_df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/mpi-ge_aa_ann.csv', index_col=0)
    aa_ratios=mpi_df.loc[year].values.transpose().squeeze()
elif model=='CanESM5':
    df = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6/cmip6_aa_ann.csv', index_col=0)
    canesm_col = [s for s in df.columns if 'CanESM5'.rstrip() in s]
    aa_ratios = df[canesm_col].loc[year].values.squeeze().ravel()
else:
    print('Model not valid!')
    sys.exit()



## observed AA; 
AA = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa_ann.csv', index_col=0)
a = AA.loc[year].mean()


distribution_list = []


if model=='CMIP6':
    
    for i in np.arange(0,n_times):
    
        # sample with replacement of N models
        N = random.randrange(2, np.shape(models)[0])
        sample = random.choices(list(models.Name.values), k=N)
        oneruns = []
        for m in sample:
            matching = [s for s in all_realizations if m.rstrip() in s]
            n = random.randrange(0,len(matching))
            onerun = aa_ratios[matching[n]]
            oneruns.append(onerun.values)
        b = np.mean(oneruns)
            
    
        # single model
        single_model_i = random.randrange(0, np.shape(models_with_multiple_real)[0])
        single_model = models_with_multiple_real.iloc[single_model_i]
        matching = [s for s in all_realizations if single_model.Name.rstrip() in s]
        n = len(matching)
        
        # single member from the single model
        random_run_i = random.randrange(0, len(matching))
        single_run_name = matching[random_run_i]
        single_run = aa_ratios[single_run_name].values
        
        # mean of the AA ratios of that model's ensemble
        ensemble_mean = aa_ratios[matching].mean(axis=1).values
        
        # the difference of AA between the single member and the mean
        c = single_run - ensemble_mean
        
        # to compensate the loss of variance due to small size of the model's ensemble, multiply by the factor
        multiplication_factor = (n/(n-1))**0.5
    
        c = c*multiplication_factor
    
        distribution_list.append(a-b+c)
else:
    for i in np.arange(0,n_times):
    
        # sample of N runs with replacement
        N = random.randrange(2, np.shape(aa_ratios)[0])
        indices = random.choices(np.arange(0,len(aa_ratios)), k=N)
        sample = aa_ratios[indices]

        b = np.mean(sample)
    
        # single run
        jth_member = random.randrange(0, np.shape(aa_ratios)[0])
        single_run = np.squeeze(aa_ratios[jth_member])
        c = b - single_run
    
        # to compensate the loss of variance due to small size of the model's ensemble, multiply by the factor
        multiplication_factor = (N/(N-1))**0.5
    
        c = c*multiplication_factor
    
        distribution_list.append(a-b+c)  

distribution = np.array(distribution_list)
    
# plot the distribution
fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(5,5), dpi=200, sharex=False)
plt.hist(distribution, bins=np.arange(-3,3,0.1), facecolor='darkgrey', edgecolor='k')

#print the p-value
cond = np.sum(distribution < 0)
print('P-value: '+str(np.sum(distribution < 0)/n_times))