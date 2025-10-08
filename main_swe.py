#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:45:53 2024

@author: vpremier
"""

import os
import xarray as xr
import rioxarray
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from plots import *

"""
Import the time-series for the basin and insert the information.
"""

# select the basin and the reference season
basin = 'Adige' 
# hy_xxxx = 'hy2223'
hy_xxxx = None

input_dir = r'/mnt/CEPH_PROJECTS/PROSNOW/LISFLOOD/input_data' #`os.getcwd()

# output directory
outdir = os.path.join(input_dir, basin, 'results_im')

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
elv = open_ds(os.path.join(input_dir, basin), 'elv', 'Band1')
# slope = open_ds(os.path.join(input_dir, basin), 'gradient', 'Band1')


# snow, melt and ice melt
snow, melt = get_snow_melt(pr, ta, elvstd)
ice_melt = get_ice_melt(ta)



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


# kaccum_mean.plot()

# plot_single_figure(basin, kaccum_mean)
# plt.savefig(os.path.join(outdir, f'kaccum_{basin}.png'))





#%%

"""
Compute a new snowmelt coefficient from EO data.
The coefficient is computed for each year through the function get_eo_cm. 
An average over 5 seasons (17/18 to 21/22) is then computed.  
A season (22/23) is left for evaluation purposes. 
"""

# compute the snowmelt coefficient calibrated through earth observation (EO)
# data with the method proposed by Pistocchi et al., 2017 for a specific year
# cm_eo1 = get_eo_cm(scf, snow, melt, ice_melt)

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
# cm_eo1_mean_filled.to_netcdf(os.path.join(outdir, 'cm_eo1', f'{basin}_cm_eo1_filled.nc'))

#%%


"""
Compute SWE with the traditional LISFLOOD coefficient and compare it with EO 
SCF. To convert SWE to SCF, we use two parametrization: i) Zaitchik and Rodell (2009)
and ii) Swenson et Lawrence (2012)
"""

# compute the snow water equivalent with the old coefficient
swe_l = compute_swe(snow, melt, ice_melt, cm_l).compute()

# conversion to scf
scf_l_swenson = scf_param_swenson(swe_l, elvstd, kaccum_mean)
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
# cm_eo2_mean_filled.to_netcdf(os.path.join(outdir, 'cm_eo2', f'{basin}_cm_eo2_filled.nc'))



#%%
"""
Compute SWE and SCF (with the desired parametrization) with the new coefficients.
"""

swe_eo1 = compute_swe(snow, melt, ice_melt, cm_eo1_mean_filled).compute()
scf_eo1 =  scf_param_swenson(swe_eo1, elvstd, kaccum_mean)

swe_eo2 = compute_swe(snow, melt, ice_melt, cm_eo2_mean_filled).compute()
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
# plt.savefig(os.path.join(outdir, f'SCA_{basin}.svg'))



plt.rcParams.update({
    'font.size': 20,  # Base font size
    'axes.titlesize': 20,  # Title font size
    'axes.labelsize': 20,  # X and Y label font size
    'xtick.labelsize': 20,  # X-axis tick font size
    'ytick.labelsize': 20,  # Y-axis tick font size
    'legend.fontsize': 20,  # Legend font size
    'figure.titlesize': 20  # Figure title font size
})
    

if basin == 'Adige':
    
    swe_cima = xr.open_mfdataset(
        f'/mnt/CEPH_PROJECTS/PROSNOW/LISFLOOD/SWE/CIMA/*/ITSNOW_SWE_*.nc',
        combine='nested',
        concat_dim='time'  # Replace 'time' with the actual time dimension in your dataset
    )
    
    # # Create a time coordinate array
    start = f'2016-09-01'
    end = f'2023-08-31'
    time = pd.date_range(start=start, end=end, periods=swe_cima.dims['time'])
    swe_cima = swe_cima.assign_coords({"time": time})
    
    
    # Assign CRS (geographic)
    swe_cima = swe_cima.rio.write_crs("EPSG:4326")
    # Set spatial dimensions for rioxarray
    swe_cima = swe_cima.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude")
    
    
    # Get shape of the grid
    ny, nx = swe_cima.rio.shape  # (rows, cols)
    
    # Extract bounds and resolution
    # xmin, ymin, xmax, ymax = swe_cima.rio.bounds()
    dx, dy = swe_cima.rio.resolution()
    
    # Rebuild coordinate arrays
    xmin = float(swe_cima.attrs['geospatial_lon_min'])
    xmax = float(swe_cima.attrs['geospatial_lon_max'])
    ymin = float(swe_cima.attrs['geospatial_lat_min'])
    ymax = float(swe_cima.attrs['geospatial_lat_max'])
    
    
    lons = np.linspace(xmin, xmax, nx)
    lats = np.linspace(ymax, ymin, ny)  # note ymin < ymax, so go downward
    
    # Assign them back
    swe_cima = swe_cima.assign_coords({
        "Longitude": ("x", lons),
        "Latitude": ("y", lats)
    })
    
    
    swe_cima = swe_cima.where(swe_cima["SWE"] != -9223372036854775808)
    
    # Now interpolate
    swe_resampled = swe_cima.interp(
        Longitude=pr.lon,
        Latitude=pr.lat,
        method="linear"
    )

    
    mask = (swe_l.mean(dim=['time']) >0) &  (swe_eo2.max(dim='time') < 2000)
    mask.plot()
    
    total_pixel = mask.sum()
    
    
    plt.figure(figsize=(10,5))
    swe_mean_old= swe_l.where(mask).mean(dim=['lat','lon'])
    swe_mean_cima= swe_resampled.where(mask).sum(dim=['lat','lon'])/total_pixel
    swe_mean_new = swe_eo2.where(mask).mean(dim=['lat','lon'])
    swe_mean_pist = swe_eo1.where(mask).mean(dim=['lat','lon'])
    
    
    colors = ['black', '#56B4E9', '#D55E00', '#009E73'] 
    plt.figure(figsize=(24, 3))
    plt.plot(swe_mean_cima.time, swe_mean_cima.SWE, label='IT-SNOW', color=colors[0], linewidth=3, linestyle='-')
    plt.plot(swe_mean_old.time, swe_mean_old, label ='L-C$_{m}$', color=colors[1], linewidth=3, linestyle='-')
    plt.plot(swe_mean_new.time, swe_mean_new.values, label ='EO-C$_{m}$', color=colors[2], linewidth=3, linestyle='-')
        
    
    plt.ylim(-1, 401)
    plt.ylabel("SWE [mm]")
    
    # Add grid lines for clarity
    plt.grid(alpha=0.3)
        
    # Hide x-tick labels
    ax = plt.gca()
    
    plt.xlim(swe_l.time[0], swe_l.time[-1])
    
    # Enhance layout
    plt.tight_layout()
    plt.legend()
    
    plt.savefig(r'/mnt/CEPH_PROJECTS/PROSNOW/LISFLOOD/Results/new_withIM/Adige_swe.png')

    
    
elif basin == 'Alpenrhein':
        
    # SWE OSHD
    path_oshd = r'/mnt/CEPH_PROJECTS/PROSNOW/GeoFrame/Dischma/SWE_oshd_1km/SWE_oshd_1km.tif'
    swe_oshd = xr.open_dataset(path_oshd)
    
    # Define start and end dates
    start_date = "2017-10-01"
    end_date = "2022-06-30"
    
    # Create a time coordinate array
    time = pd.date_range(start=start_date, end=end_date, periods=swe_oshd.dims['band'])
    
    # Replace 'bands' dimension with 'time' dimension
    swe_oshd = swe_oshd.rename({"band": "time"}).assign_coords({"time": time})
    
    
    # change projection 
    # Ensure the datasets have crs attributes for rioxarray
    swe_l.rio.write_crs("EPSG:4326", inplace=True)  # Replace with the correct CRS if known
    swe_eo2.rio.write_crs("EPSG:4326", inplace=True)  # Replace with the correct CRS if known
    
    swe_oshd.rio.write_crs("EPSG:2056", inplace=True)  # Replace with the correct CRS if known
    
    # Step 2: Reproject ds2 to match the CRS of ds1
    swe_rpj = swe_l.rio.reproject(swe_oshd.rio.crs)
    swe_new_rpj = swe_eo2.rio.reproject(swe_oshd.rio.crs)
    
    
    # Define the target resolution and reproject ds2 to match ds1
    target_res = swe_oshd.x[1]-swe_oshd.x[0]
    swe_resampled = swe_rpj.interp(
        x=swe_oshd.x,
        y=swe_oshd.y,
        method='nearest'
    )
    
    swe_new_resampled = swe_new_rpj.interp(
        x=swe_oshd.x,
        y=swe_oshd.y,
        method='nearest'
    )
    

    
    
    mask = (swe_oshd.mean(dim=['time'])['band_data'] >0).values
    
    swe_mean_oshd = swe_oshd.sel(time=slice(swe_resampled.time[0],swe_resampled.time[-1])).where(mask).mean(dim=['x','y'])['band_data']
    swe_mean_old = swe_resampled.where(mask).mean(dim=['x','y'])
    swe_mean_new = swe_new_resampled.where(mask).mean(dim=['x','y'])
    
    
    
    colors = ['black', '#56B4E9', '#D55E00', '#009E73'] 
    plt.figure(figsize=(24, 3))
    plt.plot(swe_mean_oshd.time, swe_mean_oshd, label='OSHD', color=colors[0], linewidth=3, linestyle='-')
    plt.plot(swe_mean_old.time, swe_mean_old, label ='L-C$_{m}$', color=colors[1], linewidth=3, linestyle='-')
    plt.plot(swe_mean_new.time, swe_mean_new.values, label ='EO-C$_{m}$', color=colors[2], linewidth=3, linestyle='-')
        
    
    plt.ylim(-1,700)
    plt.ylabel("SWE [mm]")
    
    # Add grid lines for clarity
    plt.grid(alpha=0.3)
        
    # Hide x-tick labels
    ax = plt.gca()
    
    plt.xlim(swe_mean_oshd.time[0], swe_mean_oshd.time[-1])
    
    # Enhance layout
    plt.tight_layout()
    
    plt.legend()

    
    plt.savefig(r'/mnt/CEPH_PROJECTS/PROSNOW/LISFLOOD/Results/Figures/SWE/Dischma_swe.png')




# Function to compute pixel-wise average bias, RMSE, and correlation over time
def compute_pixelwise_statistics(modelled, target, mask):
    """
    Compute the average bias, RMSE, and correlation for each pixel over time.
    
    Parameters:
        modelled (xr.DataArray): Modelled data (e.g., scf_lisflood) with dimensions (time, lat, lon).
        target (xr.DataArray): Target data (e.g., sca_corr) with dimensions (time, lat, lon).
        mask (2D array): Mask to exclude invalid pixels (1 for masked, 0 for valid).
    
    Returns:
        avg_bias (xr.DataArray): Time-averaged pixel-wise bias with dimensions (lat, lon).
        avg_rmse (xr.DataArray): Time-averaged pixel-wise RMSE with dimensions (lat, lon).
        correlation (xr.DataArray): Pixel-wise correlation over time with dimensions (lat, lon).
    """
    # Apply mask to exclude invalid pixels
    mask_da = xr.DataArray(mask, coords=[modelled.lat, modelled.lon], dims=["lat", "lon"])
    mask_broadcasted = mask_da.expand_dims(time=modelled.time)

    # Identify grid points where both datasets are zero
    nonzero_condition = (modelled != 0) | (target != 0)  # True when at least one is non-zero

    # Combine the non-zero condition with the mask
    valid_condition = ~mask_broadcasted & nonzero_condition
    
    # Calculate pixel-wise bias
    pixel_bias = (modelled - target).where(valid_condition)
    avg_bias = pixel_bias.mean(dim="time", skipna=True)  # Average bias over time for each pixel
    
    # Calculate pixel-wise RMSE
    pixel_squared_error = ((modelled - target) ** 2).where(valid_condition)
    avg_rmse = np.sqrt(pixel_squared_error.mean(dim="time", skipna=True))  # Time-averaged RMSE

    # Calculate pixel-wise correlation
    modelled_mean = modelled.where(valid_condition).mean(dim="time", skipna=True)
    target_mean = target.where(valid_condition).mean(dim="time", skipna=True)
    modelled_anomaly = modelled - modelled_mean
    target_anomaly = target - target_mean

    numerator = (modelled_anomaly * target_anomaly).where(valid_condition).sum(dim="time", skipna=True)
    denominator = np.sqrt(
        (modelled_anomaly ** 2).where(valid_condition).sum(dim="time", skipna=True) *
        (target_anomaly ** 2).where(valid_condition).sum(dim="time", skipna=True)
    )
    correlation = numerator / denominator

    return avg_bias, avg_rmse, correlation


bias_param1, rmse_param1, corr_param1 = compute_pixelwise_statistics(swe_l, swe_resampled, ~mask)
bias_param2, rmse_param2, corr_param2 = compute_pixelwise_statistics(swe_eo2, swe_resampled, ~mask)
# bias_param3, rmse_param3, corr_param3 = compute_pixelwise_statistics(swe_eo1, swe_resampled.SWE, ~mask)

bias_param1, rmse_param1, corr_param1 = compute_pixelwise_statistics(swe_resampled, swe_oshd, ~mask)
bias_param2, rmse_param2, corr_param2 = compute_pixelwise_statistics(swe_new_resampled, swe_oshd, ~mask)
# bias_param3, rmse_param3, corr_param3 = compute_pixelwise_statistics(swe_pist_resampled, swe_oshd, ~mask)



print('BIAS LISFLOOD=%.2f' % np.nanmean(bias_param1.values))
print('RMSE Lisflood=%.2f' % np.nanmean(rmse_param1.values))
print('corr Lisflood=%.2f' % np.nanmean(corr_param1.values))

print('BIAS OPT=%.2f' % np.nanmean(bias_param2.values))
print('RMSE OPT=%.2f' % np.nanmean(rmse_param2.values))
print('corr OPT=%.2f' % np.nanmean(corr_param2.values))

print('BIAS Pistocchi=%.2f' % np.nanmean(bias_param3.values))
print('RMSE Pistocchi=%.2f' % np.nanmean(rmse_param3.values))
print('corr Pistocchi=%.2f' % np.nanmean(corr_param3.values))
