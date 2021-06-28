# Arctic amplification

This repository contains python- and R-scritps to calculate Arctic amplification metrics and figures

## Post process the observational data
With these instructions you can post-process the observational datasets to calculate the Arctic amplification diagnostics (Rantanen et al. 2021)


### 1. Download manually each observational dataset from their sources
BEST: http://berkeleyearth.lbl.gov/auto/Global/Gridded/Land_and_Ocean_LatLong1.nc
COWTAN: https://www-users.york.ac.uk/~kdc3/papers/coverage2013/had4_krig_v2_0_0.nc.gz
GISTEMP: https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz
ERA5 1950-1978: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means-preliminary-back-extension?tab=overview
ERA5 1979-2019: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

### 2. Merge ERA5 into one file
cdo -mergetime input1 input2 output


### 3. Regrid the datasets into 0.5° horizontal resolution (this will take couple of hours for all the datasets!):
cdo -remapcon,r720x360 input output

### 4. Calculate anomalies relative to some baseline (1981-2010) period
a. Calculate climatology
cdo -ymonmean -selyear,1981/2010 input input-clim
b. Subtract the climatology to get the departures 
cdo -sub input test-clim test-anom


### 5. Fix Berkeley Earth data time axis
python fix_berkeley_earth_timeaxis.py


### The datasets undergone these steps so far are availabe from here: https://fmi100-my.sharepoint.com/:f:/g/personal/mika_rantanen_fmi_fi/Eh57LkCaG1pAqOF1I1pRmzcB39CW13XwkekxLeA7ZZZxmA?e=bbN4Z3

### 6. Calculate average of the observational datasets
cdo -b F32  -ensmean -selyear,1980/2019 -selvar,temperature BEST-regridded-retimed-anom.nc -selyear,1980/2019 -selvar,temperature_anomaly COWTAN-regridded-anom.nc -selyear,1980/2019 -selvar,tempanomaly GISTEMP-regridded-anom.nc -selyear,1980/2019 -selvar,t2m ERA5-regridded-anom.nc GBWE.nc


file "GBWE.nc" is the average of the four observational datasets


## The bootstrap confidence intervals

Run `calc_bootstrapCI_temps_obs.R`.
Inputs `arctic_temps_obs.csv` and `reference_temps_obs.csv`,
output `bootstrapCI_temps_obs.csv`.
