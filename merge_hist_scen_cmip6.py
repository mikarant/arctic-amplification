import numpy as np
import pandas as pd
import xarray as xr
import glob
#from cdo import *
#cdo= Cdo()




models = pd.read_excel('/users/mprantan/python/arctic/list_of_models.xlsx')

scen = 'ssp245'

# define the datapaths for the original model data
hist_path = '/fmi/projappl/project_2001961/cmip6/tas/historical/'

scenario_path = '/fmi/projappl/project_2001961/cmip6/tas/'+scen+'/'

# define path for merged data
outfilepath = '/fmi/projappl/project_2003992/cmip6/xr_merged_ssp245/'


number_of_models = len(models)

# loop over each model
for i in np.arange(0, number_of_models):
    
    m = models.iloc[i]
#    N = m.N
    
    
    # extract historical runs of the given model
    hist_realizations = hist_path + '*' + m.Name.rstrip() + '_historical*'
    hist_realization_names = glob.glob(hist_realizations)
    
    # extract scenario runs
    scen_realizations = scenario_path + '*' + m.Name.rstrip() + '_'+scen+'*'
    scen_realization_names = glob.glob(scen_realizations)
    
    

    # loop over the realizations
    for j in range(0, len(hist_realization_names)):
        
        # find out the realization identifier
        identifier = hist_realization_names[j].split('_')[-3]
        
        
        print(m.Name, identifier)
        
        # check if corresponding scenario run exist
        scen_cond = scenario_path + '*' + m.Name.rstrip() + '_'+scen+'*'+identifier+'*'+'201501-'+'*'
        
        scen_full_name = glob.glob(scen_cond)
        
        # if it exists, merge historical and scenario runs 
        if scen_full_name:
            print(hist_realization_names[j], flush=True)
            print(scen_full_name[0], flush=True)
            
            
            outfile = outfilepath + m.Name.rstrip() + '_' +identifier+'.nc'
            print(outfile)
            
            da_hist = xr.open_dataset(hist_realization_names[j]).tas
            da_scen = xr.open_dataset(scen_full_name[0]).tas
            
            # merge runs and calculate annual means
            da_merge = xr.concat([da_hist, da_scen], dim='time').groupby('time.year').mean('time')
            da_merge = da_merge.rename({'year':'time'})
            
            ds_merge = da_merge.to_dataset()
            
            # output the merged run into netcdf
            ds_merge.to_netcdf(outfile)
                        
 
            
        
    

    



