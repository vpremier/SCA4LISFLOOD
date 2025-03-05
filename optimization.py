#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:57:50 2024

@author: vpremier
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dask.diagnostics import ProgressBar


from utils import *


"""
Import the time-series for the basin and insert the information.
"""

# select the basin and the reference season
basin = 'Salzach' 
hy_xxxx = 'hy1819'

input_dir = r'/mnt/CEPH_PROJECTS/PROSNOW/LISFLOOD/input_data'

# output directory
outdir = os.path.join(input_dir, basin, 'results')

# paths to the time-series of sca
path_scf = os.path.join(input_dir, basin, 'SCF')


nc_name = os.path.join(path_scf , f'{basin}_{hy_xxxx}.nc')


# load the SCA time-series
scf = xr.open_dataset(nc_name).SCF.load()
scf = scf.transpose('time', 'lat', 'lon')

# get date start and end
date_start = scf.time[0].values
date_end = scf.time[-1].values


# temperature and precipitation time-series
pr = open_ds(os.path.join(input_dir, basin), 'pr', 'pr6', date_start, date_end)
ta = open_ds(os.path.join(input_dir, basin), 'ta', 'ta6', date_start, date_end)


# traditional snowmelt coefficient resulting from the hydrological calibration 
# of the LISFLOOD model

cm_l = open_ds(os.path.join(input_dir, basin), 'SnowMelt', 'SnowMeltCoef')



# auxiliary information
elvstd = open_ds(os.path.join(input_dir, basin), 'elvstd', 'Band1')
kaccum = xr.open_dataset(os.path.join(input_dir, basin, f'results/kaccum/{basin}_kaccum.nc')).kaccum.values

# snow and melt
snow, melt = get_snow_melt(pr, ta, elvstd)
snow = snow.load()
melt = melt.load()



def objective_function(cm, snow, melt, kaccum, elvstd, time, observed_scf):
    """
    Objective function to optimize a parameter `cm` for snow cover fraction (SCF) modeling.
    
    The function simulates snow water equivalent (SWE) and SCF over time and calculates 
    the mean squared error between the predicted SCF and observed SCF.

    Parameters:
    ----------
    cm : float
        Parameter to be optimized, affecting the melt rate.
    snow : array-like
        Snowfall values for each time step.
    melt : array-like
        Snowmelt values for each time step.
    kaccum : float
        Accumulation factor used in the SCF calculation.
    elvstd : float or array-like
        Standard deviation of elevation used to calculate Nmelt.
    time : array-like
        Time series corresponding to snowfall and melt data.
    observed_scf : array-like
        Observed snow cover fraction to compare with predictions.

    Returns:
    -------
    float
        Mean squared error (MSE) between the predicted SCF and the observed SCF.
    """

    # Compute the melt factor Nmelt, ensuring it's never below a threshold to avoid division issues
    max_topo = np.maximum(10, elvstd)
    Nmelt = 200 / max_topo  # Controls the rate of SCF decrease with melting

    # Initialize variables
    scf = 0  # Snow cover fraction (SCF)
    fsn = 0  # Previous SCF value
    scfList = []  # Stores SCF values over time
    int_snow = 0  # Integrated snow amount
    swe = []  # Snow water equivalent values  
    
    # Iterate over the time series
    for i, t in enumerate(time):
        if i == 0:
            swe.append(0)  # No snow at the beginning
            scfList.append(0)  # Initial SCF is zero
        else:
            # Compute the day of the year (DOY)
            doy = int(pd.to_datetime(t).strftime('%Y%j')[-3:])

            # Seasonal coefficient (varies throughout the year)
            c_seas = 0.5 * np.sin((2 * np.pi / 365.25) * (doy - 81))

            # Compute new SWE (previous SWE + snowfall - melt)
            swe_values = swe[i-1] + snow[i] - (cm + c_seas) * melt[i]
            swe_values[swe_values < 0] = 0  # Ensure SWE is not negative
            swe.append(swe_values)

            # Current and previous SWE values
            swe_curr = swe[i]
            swe_prev = swe[i-1]

            # Delta SWE (change in SWE)
            delta_swe = swe_curr - swe_prev

            # No snow condition
            if swe_curr == 0:
                scf = 0  # No snow cover fraction
                int_snow = 0  # Reset integrated snow accumulation

            # Accumulation phase (new snowfall increases SWE)
            elif delta_swe > 0:
                s = np.tanh(kaccum * delta_swe)  # Accumulation factor
                scf = 1 - ((1 - s) * (1 - fsn))  # Update SCF using previous SCF value

                # Update integrated snow
                temp_intsnow = swe_curr / (0.5 * (np.cos(np.pi * (1 - max(1e-6, fsn)) ** (1 / Nmelt)) + 1))
                int_snow = np.minimum(temp_intsnow, 1e8)  # Prevent overflow with a max cap

            # Melting phase (SWE decreases)
            elif delta_swe < 0:
                # Compute maximum SWE
                Wmax = int_snow
                Smr = np.minimum(1, swe_curr / Wmax)
                scf = (1 - ((1 / np.pi) * np.arccos(2 * Smr - 1)) ** Nmelt)

            # Store the SCF value for this time step
            fsn = scf
            scfList.append(float(scf) * 100)  # Convert to percentage

    # Compute the mean squared error (MSE) between predicted SCF and observed SCF
    predicted_scf = np.array(scfList)
    error = np.sum((predicted_scf - observed_scf) ** 2)

    return error




def optimize_cm(snow, melt, kaccum, elvstd, time, observed_scf):
    """
    Optimize the `cm` parameter for a given pixel using the objective function.

    This function minimizes the error between observed and predicted snow cover 
    fraction (SCF) by optimizing the melt factor `cm`.
    """

    # Define the objective function wrapper to optimize `cm`
    def objective(cm):
        return objective_function(cm, snow, melt, kaccum, elvstd, time, observed_scf)

    # Initial guess for `cm`
    initial_guess = [1.0]

    # Perform optimization using the L-BFGS-B method, with `cm` constrained between 0.5 and 10
    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=[(0.5, 10)])

    # Return the optimized `cm` value
    return result.x[0]  



# Ensure that your datasets are chunked for Dask
snow = snow.chunk({'lat': 10, 'lon': 10, 'time': -1})  # Adjust chunk size based on your memory capacity
melt = melt.chunk({'lat': 10, 'lon': 10, 'time': -1})
scf = scf.chunk({'lat': 10, 'lon': 10, 'time': -1})



# Optimize `cm` for each pixel, including time as an argument
optimized_cm = xr.apply_ufunc(
    optimize_cm,
    snow, melt, kaccum, elvstd, scf.time, scf,  # xarray datasets
    input_core_dims=[['time'], ['time'], [], [], ['time'], ['time']],  # Time dimension is reduced
    vectorize=True,  # Broadcast over lat and lon
    dask='parallelized',  # Enable Dask parallelization
    output_dtypes=[np.float64]  # Specify output dtype
)

# Apply some masks: exclude when there was no snow
mean_scf = scf.mean(dim="time").values
optimized_cm = optimized_cm.where(~np.isnan(mean_scf)) 
optimized_cm = optimized_cm.where(mean_scf>0) 
optimized_cm = optimized_cm.where(~np.isnan(cm_l))  # Apply mask

# Set attributes
optimized_cm = optimized_cm.assign_coords(cm_l.coords)  # Copy coordinates
optimized_cm.attrs.update(cm_l.attrs)  # Copy attributes including CRS if available

# Set variable name and attributes
optimized_cm.name = "SnowMeltCoef"
optimized_cm.attrs = {
    "long_name": "SnowMeltCoef",
    "units": "mm/(°C day)",
    "description": "Snowmelt coefficient estimated based on EO snow cover fraction data.",
    "reference": "Premier, V., Moschini, F., Casado-Rodríguez, J., Bavera, D., Marin, C., Pistocchi, A., (in preparation). Technical note: Assessing the Impact of Earth Observation Data-Driven Calibration of the Melting Coefficient on the LISFLOOD Snow Module."
}

    
outpath = os.path.join(outdir, 'cm_eo2', f'{basin}_{hy_xxxx}_cm_eo2.nc')

# Use a Dask progress bar to monitor progress
with ProgressBar():
    optimized_cm.to_netcdf(outpath)

