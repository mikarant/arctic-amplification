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
from cartopy.util import add_cyclic_point
from scipy import stats


def plotMap():

    #Set the projection information
    proj = ccrs.NorthPolarStereo()
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

datapath = '/Users/rantanem/Documents/python/data/arctic_warming/'

files = {'gistemp': datapath + 'GISTEMP.nc',
         'best': datapath + 'BEST-retimed.nc',
         'hadcrut': datapath + 'hadcrut5.nc',
         'era5' : datapath + 'era5_t2m_1950-2021.nc',
         }

variables = {'gistemp': 'tempanomaly',
             'best': 'temperature',
             'hadcrut': 'tas_mean',
             'era5': 't2m',
             }

lons = {'gistemp': 'lon',
        'best': 'longitude',
        'hadcrut': 'longitude',
        'era5': 'longitude',
        }

lats = {'gistemp': 'lat',
        'best': 'latitude',
        'hadcrut': 'latitude',
        'era5': 'latitude',
        }

titles = {'gistemp': 'a) Gistemp',
        'best': 'b) Berkeley Earth',
        'hadcrut': 'c) HadCRUT5',
        'era5': 'd) ERA5',
        }
trendfiles = {}

# define the starting and ending years of the trend
syear = 1979
eyear = 2021
years = np.arange(syear, eyear+1, 1)

for ff in files:
    print('Calculating trends for '+ff)
    # calculate annual means
    annual_means = cdo.yearmean(input =  "-selyear,"+str(syear)+"/"+str(eyear)+" -sellonlatbox,0,360,45,90 " + files[ff])

    # open the dataset
    ds = xr.open_dataset(annual_means)
    
    f = ds[variables[ff]]
    
    # allocate data arrays
    slopes = np.zeros((len(f[lats[ff]]), len(f[lons[ff]])))
    pvals = np.zeros((len(f[lats[ff]]), len(f[lons[ff]])))
    
    jj=0
    for j in range(0,len(f[lats[ff]])):
        ii=0
        for i in range(0,len(f[lons[ff]])):
            vals = f[:,j,i].values
            # calculate linear trends at each grid point
            slope, _, _, p_value, _ = stats.linregress(years, vals)
            slopes[j,i] = slope
            pvals[j,i] = p_value
            ii += 1
        jj += 1
    
    # mask non-significant trends
    mask = pvals < 0.05
    slopes[~mask] = np.nan
    
    slopes_da = xr.DataArray(slopes, coords=[f[lats[ff]], f[lons[ff]]], dims=["lat", "lon"])


    trendfiles[ff] = slopes_da
    

cmin = -1.5
cmax=  1.5
inc = 0.25
newcm = 'RdYlBu_r'

f_levels = np.arange(cmin,cmax+inc,inc)

#Get a new background map figure
fig, axlist = plotMap()
    
I = 0
for ff in trendfiles:
    
    print('Plotting '+ff)
    ds = trendfiles[ff]
    
    # multiply trend by 10 to get decadal trend
    s = ds*10

    
    f_new, new_lon = add_cyclic_point(s, coord=ds['lon'])

    # plot the field
    f_contourf = axlist[I].contourf(new_lon, ds['lat'], f_new, levels=f_levels, zorder=2, extend='both', 
                            cmap=newcm, transform = ccrs.PlateCarree())


    axlist[I].set_title(titles[ff], fontsize=16)
    I += 1


fig.subplots_adjust(right=0.8, top=0.75, wspace=0.1)
cbar_ax = fig.add_axes([0.15, 0.1, 0.62, 0.05])
cb = fig.colorbar(f_contourf, orientation='horizontal',pad=0.05,fraction=0.053, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
cb.set_label(label='Temperature trend [°C decade⁻¹]',fontsize=16)
labels = np.arange(cmin,cmax+inc,inc*2)
cb.set_ticks(labels)



plt.savefig('/Users/rantanem/Documents/python/figures/arctic_trends_supplementary.png',dpi=200,bbox_inches='tight')

