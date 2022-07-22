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
from matplotlib import cm  
from matplotlib.colors import ListedColormap  
from cartopy.util import add_cyclic_point


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

# define the datafiles

files = {'gistemp': datapath + 'GISTEMP.nc',
         'best': datapath + 'BEST-retimed.nc',
         'hadcrut': datapath + 'hadcrut5.nc',
         'era5' : datapath + 'era5_t2m_1950-2021.nc'}

variables = {'gistemp': 'tempanomaly',
             'best': 'temperature',
             'hadcrut': 'tas_mean',
             'era5': 't2m',
             }

lons = {'gistemp': 'lon',
        'best': 'longitude',
        'hadcrut':'longitude',
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


# calculate AA ratios and store them to "trendfiles"

for f in files:
    print('Calculating AA ratio for '+ f)

    # calculate annual means
    annual_means = cdo.yearmean(input =  "-selyear,1979/2021 " + files[f])

    # calculate trends
    trend1, trend2 = cdo.trend(input = annual_means)
        

    # take the global mean
    output = cdo.fldmean(input = trend2)

    # open the dataset and pick up the global average trend
    ds = xr.open_dataset(output)
    global_mean = ds[variables[f]].values.squeeze()
    
    # divide the global trend by its average
    warming_ratio = cdo.divc(global_mean, input = trend2)
    trendfiles[f] = warming_ratio
    



# define the colormap
Reds = cm.get_cmap('Reds', 12)
Blues = cm.get_cmap('Blues', 10)
newcolors = Reds(np.linspace(0, 1, 12))
# newcolors[0, :] = Blues(0.9)
newcolors[0, :] = Blues(0.7)
newcolors[1, :] = Blues(0.2)
for ii in range(2, 12, 2):
    newcolors[ii, :] = newcolors[ii + 1, :]
newcolors = np.vstack((newcolors, newcolors[-1], newcolors[-1]))
newcolors[-2:, :3] = newcolors[-2:, :3] * 0.4
newcm = ListedColormap(newcolors)
newcm.set_under(cm.get_cmap('Blues')(0.99), 1.0)
newcm.set_over(cm.get_cmap('Greys')(0.75), 1.0)


cmin = 0
cmax=  7
inc = 0.5

f_levels = np.arange(cmin,cmax+inc,inc)

#Get a new background map figure
fig, axlist = plotMap()
    
i = 0
for ff in trendfiles:
    
    # open the datafile
    ds = xr.open_dataset(trendfiles[ff])
    
    f = ds[variables[ff]]
    
    f_new, new_lon = add_cyclic_point(f, coord=f[lons[ff]])

    # draw filled contours
    f_contourf = axlist[i].contourf(new_lon, f[lats[ff]], f_new.squeeze(), levels=f_levels, zorder=2, extend='both', 
                            cmap=newcm, transform = ccrs.PlateCarree())

    # plot titles
    axlist[i].set_title(titles[ff], fontsize=16)
    i += 1


fig.subplots_adjust(right=0.8, top=0.75, wspace=0.1)
cbar_ax = fig.add_axes([0.15, 0.1, 0.62, 0.05])
cb = fig.colorbar(f_contourf, orientation='horizontal',pad=0.05,fraction=0.053, cax=cbar_ax)
cb.ax.tick_params(labelsize=16)
cb.set_label(label='Arctic amplification',fontsize=16)
labels = np.arange(cmin,cmax+inc,inc*2)
cb.set_ticks(labels)


plt.savefig('/Users/rantanem/Documents/python/figures/arctic_aa_supplementary.png',dpi=200,bbox_inches='tight')

