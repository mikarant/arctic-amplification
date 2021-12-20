#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:27:49 2021

@author: rantanem
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ds = pd.read_excel('/home/rantanem/Documents/python/data/arctic_warming/monthly_aa.xlsx', index_col=0,
                   engine = 'openpyxl')

months = ds.iloc[0,:].index
tit = []
import calendar
for X in months:
    tit.append(calendar.month_name[X][:3])

# tit.append('Average of the months')


xtit = list(ds.index)


sns.set(font_scale=1.2)
fig = plt.figure(figsize=(8,5),dpi=200)
ax = sns.heatmap(ds, annot=True, fmt=".1f", linewidths=.5, cmap =sns.cm.rocket_r,
                 cbar = False, xticklabels = tit,
                 yticklabels=xtit, annot_kws={"size": 16})
ax.tick_params(axis='both', which='both', length=0)
ax.xaxis.tick_top() # x axis on top
ax.xaxis.set_label_position('top')
# ax.set_title('Arctic warming / global warming ratio', 
             # fontweight='bold', pad=15,loc='center')
# ax.annotate('66.5°N',(0.88,0.85),xycoords='figure fraction')
plt.yticks(rotation=0) 
# plt.xlabel('Starting year of the linear trend', fontsize = 14, labelpad=10) # x-axis label with fontsize 15

figname= '/home/rantanem/Documents/python/figures/aa_values_monthly.png'

plt.savefig(figname,dpi=200,bbox_inches='tight')