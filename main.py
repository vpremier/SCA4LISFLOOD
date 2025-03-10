#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:45:53 2024

@author: vpremier
"""

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from plots import *

"""
Import the time-series for the basin and insert the information.
"""

# select the basin and the reference season
basin = 'Arve' 
# hy_xxxx = 'hy2223'
hy_xxxx = None

input_dir = os.getcwd()

# output directory
outdir = os.path.join(input_dir, basin, 'results')

# paths to the time-series of sca
path_scf = os.path.join(input_dir, basin, 'SCF')

# load the SCF time-series
if hy_xxxx:
    nc_name = os.path.join(path_scf , f'{basin}_{hy_xxxx}.nc')
    scf = xr.open_dataset(nc_name).SCF.load()
else:
    scf = xr.open_mfdataset(os.path.join(path_scf , f'{basin}_*.nc')).SCF.load()
    
scf = scf.transpose('time', 'lat', 'lon')
scf = scf.sortby("time")


# get date start and end
date_start = scf.time[0].values
date_end = scf.time[-1].values


# temperature and precipitation time-series
pr = open_ds(os.path.join(input_dir, basin), 'pr', 'pr6', date_start, date_end)
ta = open_ds(os.path.join(input_dir, basin), 'ta', 'ta6', date_start, date_end)


# traditional snowmelt coefficient resulting from the hydrological calibration 
# of the LISFLOOD model
cm_l = open_ds(os.path.join(input_dir, basin), 'SnowMeltCoef', 'SnowMeltCoef')


# auxiliary information
elvstd = open_ds(os.path.join(input_dir, basin), 'elvstd', 'Band1')
forest = open_ds(os.path.join(input_dir, basin), 'fracforest', 'Band1')
# elv = open_ds(os.path.join(input_dir, basin), 'elv', 'Band1')
# slope = open_ds(os.path.join(input_dir, basin), 'gradient', 'Band1')


# snow and melt
snow, melt = get_snow_melt(pr, ta, elvstd)


#%%

"""
Compute kaccum from EO data.
The coefficient is computed for each year through the function get_kaccum. 
An average over 5 seasons (17/18 to 21/22) is then computed.  
A season (22/23) is left independent for evaluation purposes. 
"""

# get kaccum to be used in the Swenson parametrization for retrieving scf from swe
# similarly to cm, the constant is retrieved for each season and an average 
# is then computed
# kaccum = get_kaccum(scf, snow, cm_l)
# kaccum.to_netcdf(os.path.join(outdir, 'kaccum', f'{basin}_{hy_xxxx}_kaccum.nc'))

seasons = ['1718', '1819', '1920', '2021', '2122']
kaccum_folder = os.path.join(outdir, 'kaccum')
kaccum_mean = get_mean_coeff(kaccum_folder, seasons, cm_l)
# kaccum_mean.to_netcdf(os.path.join(outdir, 'kaccum', f'{basin}_kaccum.nc'))


#%%

"""
Compute a new snowmelt coefficient from EO data.
The coefficient is computed for each year through the function get_eo_cm. 
An average over 5 seasons (17/18 to 21/22) is then computed.  
A season (22/23) is left for evaluation purposes. 
"""

# compute the snowmelt coefficient calibrated through earth observation (EO)
# data with the method proposed by Pistocchi et al., 2017 for a specific year
# cm_eo1 = get_eo_cm(scf, snow, melt)

# save the coefficient specifically calibrated for the season
# cm_eo1.to_netcdf(os.path.join(outdir, 'cm_eo1', f'{basin}_{hy_xxxx}_cm_eo1.nc'))

# retrieve a mean EO based coefficient (we use here a mean obtained for 
# five seasons (17/18 - 21/22)  
cm_folder = os.path.join(outdir, 'cm_eo1')
cm_eo1_mean = get_mean_coeff(cm_folder, seasons, cm_l)
# cm_eo1_mean.to_netcdf(os.path.join(outdir, 'cm_eo1', f'{basin}_cm_eo1.nc'))

# where the coefficient was not computed (e.g., because of missing snow) replace
# no data with the old coefficient
cm_eo1_mean_filled = cm_eo1_mean.fillna(cm_l)



#%%


"""
Compute SWE with the traditional LISFLOOD coefficient and compare it with EO 
SCF. To convert SWE to SCF, we use two parametrization: i) Zaitchik and Rodell (2009)
and ii) Swenson et Lawrence (2012)
"""

# # compute the snow water equivalent with the old coefficient
# swe_l = compute_swe(snow, melt, cm_l).compute()

# # conversion to scf
# scf_l_swenson = scf_param_swenson(swe_l, elvstd, kaccum_mean)
# scf_l_zaitchik = scf_param_zaitchik(swe_l, 4, forest) #Zaitchik and Rodell (2009)




#%%


"""
Compute a new snowmelt coefficient from EO data.
The coefficient is computed for each year through the separated script optimization.py. 
An average over 5 seasons (17/18 to 21/22) is then computed.  
A season (22/23) is left for evaluation purposes. 
"""

cm_folder = os.path.join(outdir, 'cm_eo2')
cm_eo2_mean = get_mean_coeff(cm_folder, seasons, cm_l)
# cm_eo2_mean.to_netcdf(os.path.join(outdir, 'cm_eo2', f'{basin}_cm_eo2.nc'))

# where the coefficient was not computed (e.g., because of missing snow) replace
# no data with the old coefficient
cm_eo2_mean_filled = cm_eo2_mean.fillna(cm_l)


#%%
"""
Compute SWE and SCF (with the desired parametrization) with the new coefficients.
"""

swe_eo1 = compute_swe(snow, melt, cm_eo1_mean_filled).compute()
scf_eo1 =  scf_param_swenson(swe_eo1, elvstd, kaccum_mean)

swe_eo2 = compute_swe(snow, melt, cm_eo2_mean_filled).compute()
scf_eo2 =  scf_param_swenson(swe_eo2, elvstd, kaccum_mean)


#%%
"""
Figures 2 and 3 of the manuscript

Snowmelt coefficient estimated using the hydrological calibration of 
LISFLOOD (on the left), EO data via Pistocchi et al., 2017 (in the middle), 
and EO data through the optimization approach (on the right). 
The corresponding histograms are also included.
"""

plot_snowmelt_coeff(basin, cm_l, cm_eo1_mean, cm_eo2_mean)
# plt.savefig(os.path.join(outdir, f'Cm_{basin}.png'))
 

"""
Figures 4 and 5 of the manuscript

Snowmelt coefficient estimated using the hydrological calibration of 
LISFLOOD (on the left), EO data via Pistocchi et al., 2017 (in the middle), 
and EO data through the optimization approach (on the right). 
The corresponding histograms are also included.
"""

# Compute the mean along 'time'
mean_scf = scf.mean(dim='time').values  # Convert to NumPy array

# Create a mask: True where valid (not NaN), False where invalid (NaN)
mask = np.isnan(mean_scf)

plot_sca(basin, scf, scf_l_swenson, scf_eo1, scf_eo2, mask)
# plt.savefig(os.path.join(outdir, f'SCA_{basin}.png'))

"""
Compute BIAS, RMSE and correlation
"""
print('Metrics with the parametrization of Swenson and Lawrance (2012)')
print_metrics(scf_l_swenson, scf, mask)

print('Metrics with the parametrization of Zaitchik and Rodell (2009)')
print_metrics(scf_l_zaitchik, scf, mask)

print('Metrics with the EO snowmelt coefficient obtained with Pistocchi et al. (2017)')
print_metrics(scf_eo1, scf, mask)

print('Metrics with the novel optimization method')
print_metrics(scf_eo2, scf, mask)

