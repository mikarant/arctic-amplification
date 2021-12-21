#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:48:56 2021

@author: rantanem
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


# ## annotations
annot=False

# colors and title names
cmap = plt.get_cmap("tab10")
longnames = ['CMIP5',
             'CMIP6',
             'MPI-GE',
             'CanESM5',
             ]

# uncertainty interval
range_min = 0.05
range_max = 0.95

# linewidths and thicknesses
elinewidth = 1
capthick=1

### read cmip6 results
cmip6_ref_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_global_trends.csv',index_col=0)
cmip6_arctic_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_arctic_trends.csv',index_col=0)
cmip6_ratios = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip6_aa.csv', index_col=0)


### read cmip5 results
cmip5_ref_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_global_trends.csv',index_col=0)
cmip5_arctic_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_arctic_trends.csv',index_col=0)
cmip5_ratios = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_aa.csv',index_col=0)

## read mpi results
mpi_ds = xr.open_dataset('/Users/rantanem/Documents/python/data/arctic_warming/data_pdf_plots_MPI-ESM_rcp45.nc')
mpi_ind_2019 = np.isin(np.arange(1889,2100), 2019)
mpi_ind_2018 = np.isin(np.arange(1889,2100), 2018)

## read canesm5 results
canesm_col = [s for s in cmip6_ref_trends.columns if 'CanESM5'.rstrip() in s]
canesm_ratios = cmip6_ratios[canesm_col]
canesm_arctic_trends = cmip6_arctic_trends[canesm_col]
canesm_ref_trends = cmip6_ref_trends[canesm_col]

## read miroc results
miroc_col = [s for s in cmip6_ref_trends.columns if 'MIROC-ES2L'.rstrip() in s]
miroc_ratios = cmip6_ratios[miroc_col]
miroc_arctic_trends = cmip6_arctic_trends[miroc_col]
miroc_ref_trends = cmip6_ref_trends[miroc_col]


# exclude canesm5 from cmip6
cmip6_without_canesm = [m for m in cmip6_ratios.columns  if m not in canesm_col]
cmip6_ref_trends = cmip6_ref_trends[cmip6_without_canesm]
cmip6_arctic_trends = cmip6_arctic_trends[cmip6_without_canesm]
cmip6_ratios = cmip6_ratios[cmip6_without_canesm]


### observed trends with lines
df_trends = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_trends.csv', index_col=0)
df_obs = pd.read_csv('/Users/rantanem/Documents/python/data/arctic_warming/observed_aa.csv', index_col=0)



# create the figure
xticks = np.arange(1,5)

fig, axlist= plt.subplots(nrows=1, ncols=2, figsize=(10,4), dpi=200, sharex=False)


######### CMIP5 ################
# plot cmip5 for arctic
y = cmip5_arctic_trends.loc[2019].mean(axis=0)*10
ymin = y - cmip5_arctic_trends.quantile(range_min, axis=1)[2019]*10
ymax =  cmip5_arctic_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(0.93,y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0),label='Arctic trend')

## plot cmip5 for global
y = cmip5_ref_trends.loc[2019].mean(axis=0)*10
ymin = y - cmip5_ref_trends.quantile(range_min, axis=1)[2019]*10
ymax =  cmip5_ref_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(1.07, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1),label='Global trend')




######### CMIP6 ################
## plot cmip6 for arctic
y = cmip6_arctic_trends.loc[2019].mean(axis=0)*10
ymin = y - cmip6_arctic_trends.quantile(range_min, axis=1)[2019]*10
ymax =  cmip6_arctic_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(1.93,y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

## plot cmip6 for global
y = cmip6_ref_trends.loc[2019].mean(axis=0)*10
ymin = y - cmip6_ref_trends.quantile(range_min, axis=1)[2019]*10
ymax =  cmip6_ref_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(2.07, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))



######### MPI_GE ################
## plot mpi-ge for arctic
y =  mpi_ds['arctic_trend'].values[:,mpi_ind_2019].mean(axis=0).squeeze()
ymin = y - mpi_ds['arctic_trend'][:,mpi_ind_2019].squeeze().quantile(range_min).values
ymax =  mpi_ds['arctic_trend'][:,mpi_ind_2019].squeeze().quantile(range_max,).values - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(2.93,y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

 ## plot mpi-ge for global
y = mpi_ds['global_trend'].values[:,mpi_ind_2019].mean(axis=0).squeeze()
ymin = y - mpi_ds['global_trend'][:,mpi_ind_2019].squeeze().quantile(range_min).values
ymax =  mpi_ds['global_trend'][:,mpi_ind_2019].squeeze().quantile(range_max,).values - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(3.07,y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))



######### CanESM5 ################
## plot canesm5 for arctic
y = canesm_arctic_trends.loc[2019].mean(axis=0)*10
ymin = y - canesm_arctic_trends.quantile(range_min, axis=1)[2019]*10
ymax =  canesm_arctic_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(3.93,y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

## plot canesm5 for global
y = canesm_ref_trends.loc[2019].mean(axis=0)*10
ymin = y - canesm_ref_trends.quantile(range_min, axis=1)[2019]*10
ymax =  canesm_ref_trends.quantile(range_max, axis=1)[2019]*10 - y
yerr = [[ymin],[ymax]]

axlist[0].errorbar(4.07, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))



# observed trends with lines
axlist[0].axhline(y=df_trends.loc['Global trend'].mean()*10, 
                  c=cmap(1), linestyle='--', linewidth=1)

axlist[0].axhline(y=df_trends.loc['Arctic trend'].mean()*10, 
                  c=cmap(0), linestyle='--', linewidth=1)


### define other parameters
axlist[0].grid(axis='y')
axlist[0].set_ylabel('Temperature trend\n[°C per decade]', fontsize=14)


from matplotlib import container
handles, labels = axlist[0].get_legend_handles_labels()
handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]

axlist[0].legend(handles, labels, fontsize=12, ncol=2, bbox_to_anchor=(1.02, 1.16))

axlist[0].set_xticks(xticks)
axlist[0].set_xticklabels(labels=longnames , fontsize=12)
axlist[0].set_yticks(np.arange(0,2,0.2))
axlist[0].tick_params(axis='y', which='major', labelsize=14)
axlist[0].set_ylim(0.1,1.72)



### PLOT b-plot #######


######### CMIP5 ################
## plot cmip5
y = cmip5_ratios.loc[2019].mean()
ymin = y - cmip5_ratios.quantile(range_min, axis=1)[2019]
ymax =  cmip5_ratios.quantile(range_max, axis=1)[2019] - y
yerr = [[ymin],[ymax]]


axlist[1].errorbar(1, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

y = cmip5_ratios.loc[2018].mean()
ymin = y - cmip5_ratios.quantile(range_min, axis=1)[2018]
ymax =  cmip5_ratios.quantile(range_max, axis=1)[2018] - y
yerr = [[ymin],[ymax]]


# axlist[1].errorbar(1.07, y, yerr=yerr,
#                     fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))


######### CMIP6 ################
## plot cmip6
y = cmip6_ratios.loc[2019].mean()
ymin = y - cmip6_ratios.quantile(range_min, axis=1)[2019]
ymax =  cmip6_ratios.quantile(range_max, axis=1)[2019] - y
yerr = [[ymin],[ymax]]

axlist[1].errorbar(2, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

y = cmip6_ratios.loc[2018].mean()
ymin = y - cmip6_ratios.quantile(range_min, axis=1)[2018]
ymax =  cmip6_ratios.quantile(range_max, axis=1)[2018] - y
yerr = [[ymin],[ymax]]

# axlist[1].errorbar(2.07, y, yerr=yerr,
#                     fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))

######### MPI_GE ################

## plot mpi mean & maximum
y = mpi_ds['aa'].values[:,mpi_ind_2019].mean().squeeze()
ymin = y - mpi_ds['aa'][:,mpi_ind_2019].squeeze().quantile(range_min).values
ymax =  mpi_ds['aa'][:,mpi_ind_2019].squeeze().quantile(range_max,).values - y
yerr = [[ymin],[ymax]]

axlist[1].errorbar(3, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

y = mpi_ds['aa'].values[:,mpi_ind_2018].mean().squeeze()
ymin = y - mpi_ds['aa'][:,mpi_ind_2018].squeeze().quantile(range_min).values
ymax =  mpi_ds['aa'][:,mpi_ind_2018].squeeze().quantile(range_max,).values - y
yerr = [[ymin],[ymax]]

# axlist[1].errorbar(3.07, y, yerr=yerr,
#                     fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))


######### CanESM5 ################
## plot canesm
y = canesm_ratios.loc[2019].mean()
ymin = y - canesm_ratios.quantile(range_min, axis=1)[2019]
ymax =  canesm_ratios.quantile(range_max, axis=1)[2019] - y
yerr = [[ymin],[ymax]]

axlist[1].errorbar(4, y, yerr=yerr,
                    fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(0))

y = canesm_ratios.loc[2018].mean()
ymin = y - canesm_ratios.quantile(range_min, axis=1)[2018]
ymax =  canesm_ratios.quantile(range_max, axis=1)[2018] - y
yerr = [[ymin],[ymax]]

# axlist[1].errorbar(4.07, y, yerr=yerr,
#                     fmt='o', capsize=5, capthick=2, elinewidth=elinewidth, c=cmap(1))



# observed AA with line
axlist[1].axhline(y=df_obs.loc[2019].mean(), 
                  c=cmap(0), linestyle='--', linewidth=1)


axlist[1].grid(axis='y')
axlist[1].set_ylabel('Arctic amplification', fontsize=14)

labels = ['AA 1980-2019']
axlist[1].legend(handles[:], labels[:], fontsize=12, ncol=2, bbox_to_anchor=(1.02, 1.16))

plt.xticks(xticks,labels=longnames, fontsize=12)
plt.yticks(np.arange(1.,5,0.5), fontsize=14)
axlist[1].set_ylim(1.5,4)

# increase width between the plots
plt.subplots_adjust(wspace=0.4)

## add a and b labels
axlist[0].annotate('a)', xy=(-0.30, 1), xycoords='axes fraction', fontsize=16, fontweight='bold')
axlist[1].annotate('b)', xy=(-0.27, 1), xycoords='axes fraction', fontsize=16, fontweight='bold')

figurePath = '/Users/rantanem/Documents/python/figures/'
figureName = 'simulated_trends_aa.png'
  
plt.savefig(figurePath + figureName,dpi=200,bbox_inches='tight')