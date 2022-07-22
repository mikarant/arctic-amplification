# Arctic amplification

This repository contains python- and R-scripts to calculate Arctic amplification metrics and figures presented in Rantanen et al. (2022) manuscript "The Arctic has warmed nearly four times faster than the globe since 1979"

The dataset for producing the charts and graphs of the manuscript is reposited in public repository at http://doi.org/10.23728/fmi-b2share.5d81ded56e984072a5f7162a18b60cb9. The instructions to produce the data are below.

## Post process the observational data
With these instructions you can post-process the observational datasets to calculate the Arctic amplification diagnostics. You need CDO software (https://code.mpimet.mpg.de/projects/cdo) and python version 3+. You can calculate the datasets by yourself by following these steps, or just download them from links given below.


### 1. Download manually each observational dataset from their sources
BEST: http://berkeleyearth.lbl.gov/auto/Global/Gridded/Land_and_Ocean_LatLong1.nc \
HadCRUT5: https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/HadCRUT.5.0.1.0.analysis.anomalies.ensemble_mean.nc \
GISTEMP: https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz \
ERA5 1950-1978: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means-preliminary-back-extension?tab=overview \
ERA5 1979-2021: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview \

### 2. Merge ERA5 into one file
cdo -mergetime input1 input2 output


### 3. Regrid the datasets into 0.5° horizontal resolution (this will take couple of hours for all the datasets!):
cdo -remapcon,r720x360 input output

### 4. Calculate anomalies relative to some baseline (1981-2010) period
a. Calculate climatology \
cdo -ymonmean -selyear,1981/2010 input input-clim \
b. Subtract the climatology to get the departures \
cdo -sub input test-clim test-anom


### 5. Fix Berkeley Earth data time axis
python fix_berkeley_earth_timeaxis.py


### 6. Calculate average of the observational datasets
cdo -L -b F32 -ensmean -selyear,1950/2021 -selvar,temperature BEST-regridded-retimed-anom.nc -selyear,1950/2021 -selvar,tas_mean hadcrut5-regridded-anom.nc -selyear,1950/2021 -selvar,tempanomaly GISTEMP-regridded-anom.nc -selyear,1950/2021 -selvar,t2m ERA5-regridded-anom.nc OBSAVE.nc


## Calculate the observed AA ratios and trends
### 1. Calculate the observed values
Run `calculate_observed_aa_trends.py`
### 2. Caclulate the bootstrap confidence intervals
Run `calc_bootstrapCI_temps_obs.R`.
Inputs `arctic_temps_obs.csv` and `reference_temps_obs.csv`,
output `bootstrapCI_temps_obs.csv`.



## Calculate the CMIP6-simulated AA ratios and trends
### 1. Merge historical and scenario runs
Run `merge_hist_scen_cmip6.py`
### 2. Calculate area-mean temperatures in the Arctic and globally
Run `calculate_temps_cmip6.py`
### 3. Calculate AA ratios and trends 
Run `calculate_trends_and_aa_cmip6.py`



