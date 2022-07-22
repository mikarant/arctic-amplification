#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 23:41:35 2021
This script plots the figure 1 of the manuscript

@author: rantanem
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import xarray as xr
from matplotlib import cm  
from matplotlib.colors import ListedColormap
from scipy import stats 

def plot_background(ax):
    import cartopy.feature as cfeature
    import matplotlib.path as mpath    
    import matplotlib.ticker as mticker
    ax.patch.set_facecolor('w')
    ax.spines['geo'].set_edgecolor('k')
    
    ax.set_extent([-180, 180, 50,90], crs=ccrs.PlateCarree())
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

def plot_linear_trend(ax, syear, eyear, f, c, alpha):
    
    years = f.index
    s = np.where(years==syear)[0][0]
    e = np.where(years==eyear)[0][0] + 1
    
    z = np.polyfit(years[s:e],f[s:e],1)
    p1 = np.poly1d(z)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(years[s:e], f[s:e])
    
    
    ax.plot(years[s:e],p1(years[s:e]), linewidth=1.5,color=c,alpha=alpha)
    
    return r_value


datapath = '/Users/rantanem/Documents/python/data/arctic_warming/'


temp_obs_arctic = pd.read_csv(datapath + '/arctic_temps_obs.csv',index_col=0)
temp_obs_ref = pd.read_csv(datapath +'/global_temps_obs.csv',index_col=0)

ds = xr.open_dataset(datapath +'/OBSAVE_trend.nc')
trend_new, new_lon = add_cyclic_point(ds.trend_masked*10, coord=ds['lon'])
aa_new, new_lon = add_cyclic_point(ds.aa, coord=ds['lon'])

years = temp_obs_ref.index

# range for linear trends
syear=1979
eyear=2021


# colormap for AA
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



cmin = -1.5
cmax=  1.5
inc = 0.25
trend_cm = 'RdYlBu_r'

f_levels = np.arange(cmin,cmax+inc,inc)

proj = ccrs.NorthPolarStereo()

fig8 = plt.figure(figsize=(10,10),constrained_layout=False)
gs1 = fig8.add_gridspec(nrows=2, ncols=2, left=0.0, right=0.66, wspace=0.4)
f8_ax1 = fig8.add_subplot(gs1[:-1, :])
f8_ax2 = fig8.add_subplot(gs1[-1, :-1], projection= proj)
f8_ax3 = fig8.add_subplot(gs1[-1, -1], projection= proj)

f8_ax1.plot(years, temp_obs_ref.BEST, c='blue', alpha=0.2,)
f8_ax1.plot(years, temp_obs_ref.HADCRUT, c='red', alpha=0.2, )
f8_ax1.plot(years, temp_obs_ref.GISTEMP,c='orange', alpha=0.2, )
f8_ax1.plot(years, temp_obs_ref.ERA5,c='green', alpha=0.2, )

plot_linear_trend(f8_ax1, syear, eyear, temp_obs_ref.BEST, 'blue',alpha=0.2)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_ref.HADCRUT, 'red',alpha=0.2)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_ref.GISTEMP, 'orange',alpha=0.2)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_ref.ERA5, 'green',alpha=0.2)


f8_ax1.plot(years, temp_obs_arctic.BEST, c='blue', alpha=1,label='Berkeley Earth')
f8_ax1.plot(years, temp_obs_arctic.HADCRUT, c='red', alpha=1,label='HadCRUT5')
f8_ax1.plot(years, temp_obs_arctic.GISTEMP,c='orange', alpha=1,label='Gistemp')
f8_ax1.plot(years, temp_obs_arctic.ERA5,c='green', alpha=1,label='ERA5')

plot_linear_trend(f8_ax1, syear, eyear, temp_obs_arctic.BEST, 'blue',alpha=1)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_arctic.HADCRUT, 'red',alpha=1)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_arctic.GISTEMP, 'orange',alpha=1)
plot_linear_trend(f8_ax1, syear, eyear, temp_obs_arctic.ERA5, 'green',alpha=1)



f8_ax1.grid(True)
f8_ax1.set_xlim(1950,2022)
f8_ax1.tick_params(axis='both', which='major', labelsize=16)
f8_ax1.legend(fontsize=14, loc='upper left')
f8_ax1.set_ylabel('Temperature anomaly [°C]', fontsize=16)


plot_background(f8_ax2)
f_contourf = f8_ax2.contourf(new_lon, ds['lat'], trend_new, levels=f_levels, zorder=2, extend='both', 
                            cmap=trend_cm, transform = ccrs.PlateCarree())
fig8.subplots_adjust(right=0.8, top=0.75, wspace=0.05)
cbar_ax = fig8.add_axes([0.00, 0.088, 0.27, 0.02])
cb = fig8.colorbar(f_contourf, orientation='horizontal',pad=0.0,fraction=0.0, cax=cbar_ax)
cb.ax.tick_params(labelsize=14)
cb.set_label(label='Temperature trend [°C decade⁻¹]',fontsize=16, )
labels = np.arange(cmin,cmax+inc,inc*3)
cb.set_ticks(labels)



plot_background(f8_ax3)
f_contourf = f8_ax3.contourf(new_lon, ds['lat'], aa_new, levels=np.arange(0,7.5,0.5), zorder=2, extend='both', 
                            cmap=newcm, transform = ccrs.PlateCarree())
cbar_ax = fig8.add_axes([0.39, 0.088, 0.27, 0.02])
cb = fig8.colorbar(f_contourf, orientation='horizontal',pad=0.0,fraction=0.0, cax=cbar_ax)
cb.ax.tick_params(labelsize=14)
cb.set_label(label='Local amplification',fontsize=16)
labels = np.arange(0,7.5,1)
cb.set_ticks(labels)



### annotations
f8_ax1.annotate('a)',(0,1.05), xycoords='axes fraction', fontsize=17,fontweight='bold')
f8_ax2.annotate('b)',(0,0.97), xycoords='axes fraction', fontsize=17,fontweight='bold')
f8_ax3.annotate('c)',(0,0.97), xycoords='axes fraction', fontsize=17,fontweight='bold')


plt.savefig('/Users/rantanem/Documents/python/figures/figure1.pdf',dpi=300,bbox_inches='tight')


## This part of the script plots the figure used in peer-review response

fig, ax= plt.subplots(nrows=1, ncols=2, figsize=(10,4), dpi=200, sharex=False)

ax[0].plot(years, temp_obs_arctic.ERA5,c='green', alpha=1,label='ERA5')
ax[0].plot(years, temp_obs_arctic.BEST, c='blue', alpha=1,label='Berkeley Earth')
ax[0].plot(years, temp_obs_arctic.HADCRUT, c='red', alpha=1,label='HadCRUT5')
ax[0].plot(years, temp_obs_arctic.GISTEMP,c='orange', alpha=1,label='Gistemp')
ax[0].axhline(y=0.0, color='black', linestyle='-')
ax[0].legend(fontsize=12)

ax[0].set_title('Temperatures in the Arctic', loc='left', fontsize=16)
ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].set_ylabel('Temperature anomaly [°C]', fontsize=14)

ax[1].plot(years, temp_obs_arctic.ERA5 - temp_obs_arctic.BEST, c='blue', alpha=1,label='ERA5 - Berkeley Earth')
ax[1].plot(years, temp_obs_arctic.ERA5 - temp_obs_arctic.HADCRUT, c='red', alpha=1,label='ERA5 - HadCRUT5')
ax[1].plot(years, temp_obs_arctic.ERA5 - temp_obs_arctic.GISTEMP,c='orange', alpha=1,label='ERA5 - Gistemp')
ax[1].axhline(y=0.0, color='black', linestyle='-')
ax[1].legend(fontsize=12)
ax[1].set_title('Temperature difference to ERA5', loc='left', fontsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=14)
ax[1].set_ylabel('Temperature difference [°C]', fontsize=14)

fig.subplots_adjust(wspace=0.4)

