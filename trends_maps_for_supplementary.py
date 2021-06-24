#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 20:20:47 2020

@author: rantanem
"""
from cdo import *
cdo = Cdo()
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
from matplotlib import cm  
from matplotlib.colors import ListedColormap  
from cartopy.util import add_cyclic_point
import pandas as pd
from scipy import stats


def plotMap():

    #import cartopy.feature as cfeature

    #Set the projection information
    proj = ccrs.NorthPolarStereo()
    # proj = ccrs.LambertConformal(central_longitude=18.0,central_latitude=53, standard_parallels=[53])
    #Create a figure with an axes object on which we will plot. Pass the projection to that axes.
    fig, axarr = plt.subplots(nrows=1, ncols=4, figsize=(15, 6), constrained_layout=False, dpi=200,
                          subplot_kw={'projection': proj})
    axlist = axarr.flatten()
    for ax in axlist:
        plot_background(ax)
   
    return fig, axlist


def plot_background(ax):
    import cartopy.feature as cfeature
    import matplotlib.path as mpath    
    import matplotlib.ticker as mticker
    ax.patch.set_facecolor('w')
    ax.spines['geo'].set_edgecolor('k')
    
    # ax.set_extent([40, 0, 54, 70])
    ax.set_extent([-180, 180, 50,90], crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor=cfeature.COLORS['land'])
    # ax.add_feature(cfeature.LAKES.with_scale('50m'),facecolor=cfeature.COLORS['land'],zorder=1,edgecolor='k',linewidth=1.5) 
    # ax.add_feature(cfeature.BORDERS.with_scale('50m'), zorder=10)
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), zorder=10)
    

    gl = ax.gridlines(linewidth=2, color='k', alpha=0.5, linestyle='--')
    gl.n_steps = 100
    gl.ylocator = mticker.FixedLocator([66.5])
    gl.xlocator = mticker.FixedLocator([])
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)
    return ax

datapath = '/home/rantanem/Documents/python/data/arctic_warming/'

files = {'gistemp': datapath + 'GISTEMP.nc',
         'best': datapath + 'BEST.nc',
         'cw': datapath + 'COWTAN.nc',
         'era5' : datapath + 'era5_t2m_1950-2019.nc'}

variables = {'gistemp': 'tempanomaly',
             'best': 'temperature',
             'cw': 'temperature_anomaly',
             'era5': 't2m',
             }

lons = {'gistemp': 'lon',
        'best': 'longitude',
        'cw': 'longitude',
        'era5': 'longitude',
        }

lats = {'gistemp': 'lat',
        'best': 'latitude',
        'cw': 'latitude',
        'era5': 'latitude',
        }

titles = {'gistemp': 'a) Gistemp',
        'best': 'b) Berkeley Earth',
        'cw': 'c) Cowtan & Way',
        'era5': 'd) ERA5',
        }
trendfiles = {}

# years = np.arange(1979, 2020, 1)

for ff in files:
    print(ff)
    # calculate annual means
    annual_means = cdo.yearmean(input =  "-selyear,1980/2019 -sellonlatbox,0,360,45,90 " + files[ff])

    ds = xr.open_dataset(annual_means)
    
    f = ds[variables[ff]]

    years = np.arange(1980,2020,1)
    
    slopes = np.zeros((len(f[lats[ff]]), len(f[lons[ff]])))
    pvals = np.zeros((len(f[lats[ff]]), len(f[lons[ff]])))
    
    jj=0
    for j in range(0,len(f[lats[ff]])):
        ii=0
        for i in range(0,len(f[lons[ff]])):
            # vals = f.sel(lon=i, lat=j).values
            vals = f[:,j,i].values
            slope, _, _, p_value, _ = stats.linregress(years, vals)
            slopes[j,i] = slope
            pvals[j,i] = p_value
            ii += 1
        jj += 1
    
    mask = pvals < 0.05
    slopes[~mask] = np.nan
    
    slopes_da = xr.DataArray(slopes, coords=[f[lats[ff]], f[lons[ff]]], dims=["lat", "lon"])


    trendfiles[ff] = slopes_da
    
    # print(global_mean)
    

# ##  replace ratio of means by mean of ratios
# trendfiles['cmip5'] = '/home/rantanem/Documents/python/data/arctic_warming/cmip5/cmip5_mean_aa.nc'
# variables['cmip5'] = 'aa'

# mpidata = datapath + 'data_for_fig2_3.nc'
# da = xr.open_dataset(mpidata).trend/10
# trendfiles['mpi'] = da
# variables['mpi'] = 'trend'
# lons['mpi'] = 'longitude'
# lats['mpi'] = 'latitude'
# titles['mpi'] = 'd) MPI-GE mean'




# colormap
Reds = cm.get_cmap('Reds', 12)
Blues = cm.get_cmap('Blues', 10)
newcolors = Reds(np.linspace(0, 1, 12))
# newcolors[0, :] = Blues(0.9)
newcolors[0, :] = Blues(0.8)
newcolors[1, :] = Blues(0.2)
for ii in range(2, 12, 2):
    newcolors[ii, :] = newcolors[ii + 1, :]
newcolors = np.vstack((newcolors, newcolors[-1], newcolors[-1]))
newcolors[-2:, :3] = newcolors[-2:, :3] * 0.4
newcm = ListedColormap(newcolors)
newcm.set_under(cm.get_cmap('Blues')(0.9), 1.0)


# palette = copy(plt.get_cmap('Reds'))
# palette.set_under('white', 1.0)  # 1.0 represents not transparent

cmin = -1.5
cmax=  1.5
inc = 0.25
newcm = 'RdYlBu_r'

f_levels = np.arange(cmin,cmax+inc,inc)

#Get a new background map figure
fig, axlist = plotMap()
    
I = 0
for ff in trendfiles:
    
    # ds = xr.open_dataset(trendfiles[ff])
    print(ff)
    ds = trendfiles[ff]
    
    # multiply trend by 10 to get decadal trend
    s = ds*10

    
    f_new, new_lon = add_cyclic_point(s, coord=ds['lon'])

    f_contourf = axlist[I].contourf(new_lon, ds['lat'], f_new, levels=f_levels, zorder=2, extend='both', 
                            cmap=newcm, transform = ccrs.PlateCarree())
    # f_contour = axlist[i].contour(new_lon, f[lats[ff]], f_new.squeeze(), levels=f_levels, zorder=2, colors='k', 
    #                         linewidths=0.5, transform = ccrs.PlateCarree())
    # axlist[i].clabel(f_contour, f_levels, inline=True, fontsize=10, fmt='%1.1f')


    axlist[I].set_title(titles[ff], fontsize=16)
    I += 1


fig.subplots_adjust(right=0.8, top=0.75, wspace=0.1)
cbar_ax = fig.add_axes([0.15, 0.1, 0.62, 0.05])
cb = fig.colorbar(f_contourf, orientation='horizontal',pad=0.05,fraction=0.053, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
cb.set_label(label='Temperature trend [°C decade⁻¹]',fontsize=16)
labels = np.arange(cmin,cmax+inc,inc*2)
cb.set_ticks(labels)



plt.savefig('/home/rantanem/Documents/python/figures/arctic_trends_supplementary.png',dpi=200,bbox_inches='tight')

