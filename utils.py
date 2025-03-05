#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:10:18 2025

@author: vpremier
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import glob
import warnings


def open_ds(dirname, varname, dsname, date_start=None, date_end=None):
    """
    Opens and processes a dataset for a specified variable, 
    applying a spatial mask and optional temporal selection. 
    
    This function handles multiple variables differently based on their characteristics:
    - 'pr' (Precipitation) and 'ta' (Air Temperature) are resampled to daily means.
    - 'elv' (Elevation) and other variables are loaded without temporal resampling.
    - A spatial mask is applied to all variables to filter valid grid cells.

    Parameters
    ----------
    dirname : str
        Base directory containing the data.
    varname : str
        Name of the variable to load. Supported values:
            - 'pr': Precipitation (resampled to daily mean)
            - 'ta': Air temperature (resampled to daily mean)
            - 'elv': Elevation (no temporal resampling)
            - Other variables are loaded from the 'parameters' directory without resampling.
    dsname : str
        Name of the variable inside the NetCDF file to extract.
    date_start : str, optional
        Start date for temporal selection in 'YYYY-MM-DD' format. 
        Required for 'pr' and 'ta'; ignored for other variables.
    date_end : str, optional
        End date for temporal selection in 'YYYY-MM-DD' format. 
        Required for 'pr' and 'ta'; ignored for other variables.

    Returns
    -------
    xarray.DataArray
        Processed dataset with the applied spatial mask and optional daily temporal aggregation.
    """

    # Construct the directory path for the mask
    dir_forcings = os.path.join(dirname, 'forcings')

    # Determine the variable directory based on the variable name
    dir_parameters = os.path.join(dirname, 'parameters')

    # Load the spatial mask
    mask = xr.open_mfdataset(os.path.join(dir_forcings, 'my_mask.nc'))
    if 'lat' in mask.coords:
        mask = mask.sortby("lat", ascending=False)
    elif 'y' in mask.coords:
        mask = mask.sortby("y", ascending=False)


    # Load the variable dataset (single or multiple files depending on varname)    
    if varname == 'ta': 
        var = xr.open_mfdataset(os.path.join(dir_forcings, 
                                             varname + '*20[12][0126789].nc'))
        
        # correct the metadata for 2023
        ta_23 = xr.open_dataset(os.path.join(dir_forcings, 
                                             varname + '_hourly_2023.nc'))
        ta_23['wgs_1984'] = var['wgs_1984']
        ta_23 = ta_23.rename({'ta': 'ta6'})
        
        var = var.combine_first(ta_23)
        
    elif varname == 'pr':
        var = xr.open_mfdataset(os.path.join(dir_forcings, varname + '*.nc'))
        
    else:
        var = xr.open_mfdataset(os.path.join(dir_parameters, varname + '*.nc'))

    # Apply the mask where 'area' equals 1
    if "area" in mask.data_vars:
        var_xr = var[dsname].where(mask.area.values == 1)
    elif "mask" in mask.data_vars:
        var_xr = var[dsname].where(mask.mask.values == 1)
    else:
        raise ValueError("Neither 'area' nor 'mask' found in the dataset 'mask'.")


    # Process temporal data for precipitation and temperature
    if varname in ['pr', 'ta']:
        outvar = (
            var_xr
            .sortby('time')
            .resample(time='D', closed='right', label='left').mean()
            .sel(time=slice(date_start, date_end))
        )
    else:
        outvar = var_xr  # No temporal processing for other variables

    var_xr = var_xr.sortby("lat", ascending=False)
    
    return outvar



def split_prec(pr, ta, SnowFactor=1, TempSnow=1, TempMelt=1, delta_t=1):
    """
    Partition precipitation into snow and rain, and compute snowmelt.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation (mm).
    ta : xarray.DataArray
        Daily air temperature (°C).
    SnowFactor : float, optional
        Snow correction factor applied to precipitation when temperature is 
        below TempSnow. Default is 1.
    TempSnow : float, optional
        Temperature threshold (°C) below which precipitation is considered snow.
        Default is 1.
    TempMelt : float, optional
        Temperature threshold (°C) above which snow melts. Default is 1.
    delta_t : float, optional
        Time step for the melt calculation. Default is 1.

    Returns
    -------
    snow : xarray.DataArray
        Snowfall (mm).
    melt : xarray.DataArray
        Snowmelt (mm).
    """
    
    snow = pr.where(ta < TempSnow, 0) * SnowFactor
    rain = pr.where(ta >= TempSnow, 0)
    melt = ((1 + 0.01 * rain * delta_t) * (ta - TempMelt) * delta_t).where(lambda x: x >= 0, 0)
    
    return snow.fillna(0), melt.fillna(0)



def get_snow_melt(pr, ta, elvstd, SnowFactor=1, TempSnow=1, TempMelt=1, delta_t=1, 
                  TemperatureLapseRate= 0.0065, elevation_zones=True):
    """
    Calculate snowfall and snowmelt based on precipitation, temperature, 
    and other snow model parameters.

    Parameters
    ----------
    pr : xarray.DataArray
        Daily precipitation (mm).
    ta : xarray.DataArray
        Daily air temperature (°C).
    elvstd : xarray.DataArray or float
        Standard elevation difference (m) between zones.
    SnowFactor : float, optional
        Snow correction factor applied to precipitation when temperature is 
        below TempSnow. Default is 1.
    TempSnow : float, optional
        Temperature threshold (°C) below which precipitation is considered snow.
        Default is 1.
    TempMelt : float, optional
        Temperature threshold (°C) above which snow melts. Default is 1.
    delta_t : float, optional
        Time step for the melt calculation. Default is 1.
    TemperatureLapseRate : float, optional
        Temperature lapse rate (°C/m) to account for elevation gradients.
        Default value is 0.0065.
    elevation_zones : bool, optional (default=True)
        If True, the calculation is performed over three elevation zones (lower, middle, upper).
        If False, calculation is done without elevation adjustment.

    Returns
    -------
    snow : xarray.DataArray
        Snowfall (mm).
    melt : xarray.DataArray
        Snowmelt (mm).

    Notes
    -----
    - Precipitation is partitioned into snow and rain based on the temperature threshold `TempSnow`.
    - Snowmelt is calculated using a degree-day approach, adjusted for rain-on-snow events.
    - Elevation zones are considered by adjusting temperature for upper and lower zones based on `TemperatureLapseRate` and `elvstd`.

    """

    if elevation_zones:
        # Apply temperature lapse rate adjustments for elevation zones
        lapse_adjustment = 0.9674 * TemperatureLapseRate * elvstd
        snow_a, melt_a = split_prec(pr, ta - lapse_adjustment, SnowFactor=SnowFactor, 
                                    TempSnow=TempSnow, TempMelt=TempMelt, delta_t=delta_t)
        snow_b, melt_b = split_prec(pr, ta, SnowFactor=SnowFactor, TempSnow=TempSnow, 
                                    TempMelt=TempMelt, delta_t=delta_t)
        snow_c, melt_c = split_prec(pr, ta + lapse_adjustment, SnowFactor=SnowFactor, 
                                    TempSnow=TempSnow, TempMelt=TempMelt, delta_t=delta_t)

        # Average results across the three elevation zones
        snow = (snow_a + snow_b + snow_c) / 3
        melt = (melt_a + melt_b + melt_c) / 3

    else:
        # No elevation adjustment
        snow, melt = split_prec(pr, ta, SnowFactor=SnowFactor, TempSnow=TempSnow, 
                                TempMelt=TempMelt, delta_t=delta_t)

    return snow, melt



def compute_swe(snow, melt, cm):
    """
    Computes Snow Water Equivalent (SWE) based on snowfall, melt, and a melt coefficient.
    The approach reproduces the snow module of LISFLOOD and is described here
    https://ec-jrc.github.io/lisflood-model/2_04_stdLISFLOOD_snowmelt/

    Parameters
    ----------
    snow : xarray.DataArray
        Snowfall accumulation over time.
    melt : xarray.DataArray
        Melt potential over time.
    cm : xarray.DataArray
        Melt coefficient, can be spatially varying.

    Returns
    -------
    xarray.DataArray
        Computed SWE values over time.
    """

    # Initialize SWE array
    swe = xr.zeros_like(melt)

    # Convert xarray DataArrays to NumPy arrays for fast computation
    snow_array = snow.values
    melt_array = melt.values
    cm_array = cm.values
    swe_array = np.zeros_like(snow_array)

    # Compute day-of-year (DOY) in a vectorized manner
    doy = snow['time'].dt.dayofyear.data

    # Precompute seasonal coefficient (c_seas) for each day
    c_seas = 0.5 * np.sin((2 * np.pi / 365.25) * (doy - 81))

    # Iterate through time steps (starting from 1 to avoid first-step issue)
    for i in range(1, len(melt.time)):
        swe_array[i] = swe_array[i - 1] + snow_array[i] - (cm_array + c_seas[i]) * melt_array[i]
        swe_array[i] = np.maximum(swe_array[i], 0)  # Ensure SWE is non-negative
    
    # Assign computed values back to xarray structure
    swe.data = swe_array
    
    # Expand `cm` to 3D (time, lat, lon) by aligning with `snow`
    cm_3d = cm.expand_dims(dim={"time": snow.time}, axis=0)
 
    # Apply the mask: wherever cm is NaN, mask SWE as NaN too
    swe = swe.where(~cm_3d.isnull())
    
    return swe



def get_eo_cm(scf, snow, melt):
    """
    Computes the snowmelt coefficient (cm) based on snow, melt, and 
    snow-covered area (SCA), following the approach by Pistocchi et al., 2017
    that is based on the number of snow covered days.

    see https://www.mdpi.com/2073-4441/9/11/848
    
    
    Parameters
    ----------
    scf : xarray.DataArray
        Snow cover fraction (3D: time, lat, lon).
    snow : xarray.DataArray
        Snowfall accumulation over time (3D: time, lat, lon).
    melt : xarray.DataArray
        Melt potential over time (3D: time, lat, lon).

    Returns
    -------
    xarray.DataArray
        Updated melt coefficient (2D: lat, lon).
    """
    
    # Convert snow cover fraction into binary (1 = snow-covered, 0 = not snow-covered)
    scf_binary = xr.where(scf > 0, 1, 0)


    # Compute day-of-year (DOY) using xarray's built-in functionality
    doy = snow['time'].dt.dayofyear

    # Compute seasonal coefficient (c_seas) as an xarray DataArray (automatically broadcasts)
    c_seas = 0.5 * np.sin((2 * np.pi / 365.25) * (doy - 81))

    # Perform time aggregation using xarray operations
    cm = ((snow * scf_binary).sum('time') - \
              (c_seas * melt * scf_binary).sum('time')) / ((melt * scf_binary).sum('time'))

    # Set variable name and attributes
    cm.name = "SnowMeltCoef"
    cm.attrs = {
        "long_name": "SnowMeltCoef",
        "units": "mm/(°C day)",
        "description": "Snowmelt coefficient estimated based on EO snow cover fraction data.",
        "reference": "Pistocchi et al., 2017 (https://www.mdpi.com/2073-4441/9/11/848)"
    }
    
    return cm



def get_mean_coeff(folder, seasons, array_ref):
    """
    Computes a mean parameter (SnowMeltCoef or kaccum) for the specified seasons
    from NetCDF files in the given folder. The mean is calculated while
    excluding unrealistic values and missing data.

    Parameters
    ----------
    folder : str
        Path to the folder containing the NetCDF files.
    seasons : list
        List of season identifiers (e.g., ['1718', '1819', ...]).
    array_ref : xarray.DataArray
        A reference DataArray to provide the shape and coordinates for the output.

    Returns
    -------
    xarray.DataArray
        The mean snowmelt coefficient for the given seasons.
    """
    
    # Initialize a list to hold the matching file paths
    fileList = []

    # Loop through the list of seasons and find corresponding paths
    for season in seasons:
        # Construct the path with the current season
        pattern = os.path.join(folder, f'*_hy{season}_*.nc')
        # Add the found files to cm_list
        fileList.extend(glob.glob(pattern))
    
    # Initialize the coefficient array
    array = np.zeros((len(fileList), *array_ref.shape))
    
    # Load and process the cm computed for each season
    for i, path in enumerate(sorted(fileList)):        
        # Open the cm data from the NetCDF file
        ds = xr.open_dataset(path)
        
        # Get the variable name dynamically
        var_name = list(ds.data_vars.keys())[0]
        
        var = ds[var_name].values
        
        # Filter unrealistic values (apply filters to the entire array)
        if var_name == 'SnowMeltCoef':
            var = np.clip(var, 0.5, 10)  # Values must be between 0.5 and 10
        
        array[i, :, :] = var  # Assign the processed data to cm_array

    # Take the mean excluding no data values
    def compute_mean(array):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress warning
            mean = np.nanmean(array, axis=0)
        return mean
    
    mean = compute_mean(array)

    # Convert cm_mean into an xarray DataArray with appropriate coordinates
    mean_xr = xr.DataArray(mean, coords=array_ref.coords, dims=array_ref.dims)
    
    # Set variable name and attributes
    mean_xr.name = var_name
    
    if var_name == 'SnowMeltCoef':
        units = "mm/(°C day)"
        descr = "Mean snowmelt coefficient estimated based on EO snow cover fraction data."
    elif var_name == 'kaccum':
        units = "-"
        descr = "Mean constant calculated from the first accumulation."
        
    mean_xr.attrs = {
        "long_name": var_name,
        "units": units,
        "description": descr
    }
    
    return mean_xr



def scf_param_zaitchik(swe, tau, forest):
    """
    Computes the Snow Cover Fraction (SCF) using the Zaitchik and Rodell (2009) parameterization.

    Parameters
    ----------
    swe : xarray.DataArray
        Snow Water Equivalent (SWE).
    tau : float or xarray.DataArray
        Snow distribution shape parameter that relates the total amount of SWE 
        to the SCF within the pixel.
    forest : xarray.DataArray
        Fraction of forest cover (0 to 1) representing the degree of canopy.
        - `0` indicates no forest (open areas).
        - `1` indicates full forest coverage.

    Returns
    -------
    scf : xarray.DataArray
        Snow Cover Fraction (SCF) as a percentage (0 to 100).

    Notes
    -----
    - The `swe_max` threshold is defined as a linear function of forest cover, varying between 13 mm (open areas) 
      and 40 mm (dense forest). This accounts for the impact of forests on snow retention.
    - The SCF is calculated using an exponential function that incorporates `tau`, `swe`, and `swe_max`.
    - The final SCF values are capped at 100% to ensure they remain within realistic physical limits.
    
    Reference
    ---------
    Zaitchik, B. F. and Rodell, M.: Forward-looking assimilation of 
    MODIS-derived snow-covered area into a land surface model, Journal of 
    hydrometeorology, 10, 130–148, 2009.
    """
    
    # Define the maximum SWE threshold based on forest cover (varies between 13 mm and 40 mm)
    swe_max = 13 + (40 - 13) * forest  # Linearly vary between 13 and 40 based on forest cover
    
    # Compute the SCF components
    term1 = np.exp(-tau * swe / swe_max)  # First term
    term2 = (swe / swe_max) * np.exp(-tau)  # Second term
    
    # Final SCF computation
    scf = 1 - (term1 - term2)
    
    # Ensure SCF values do not exceed 1.0
    scf = np.minimum(scf, 1.0)
    
    # Convert to xarray DataArray with the same dimensions and coordinates as SWE
    scf = xr.DataArray(scf, dims=swe.dims, coords=swe.coords)
    
    # Convert SCF to percentage scale (0 to 100)
    scf = scf * 100   
    
    return scf



def scf_param_swenson(swe, elvstd, kaccum):
    """
    Convert Snow Water Equivalent (SWE) snow cover fraction (scf).
    The parametrization is based on the work by Swenson et Lawrence, 2012
    
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JD018178

    Parameters:
    -----------
    swe : xarray.DataArray
        A time series of Snow Water Equivalent (SWE) values. 

    elvstd : float
        The elevation standard deviation

    kaccum : float
        Can be a constant or can be estimated by measuring SCF and ∆W when 
        precipitation occurs over an initially snow-free area, as suggested by
        the authors

    Returns:
    --------
    scf : xarray.DataArray
        A time series of snow cover fractions [0-100%].
    """
    
    # Precompute constants
    max_topo = np.maximum(10, elvstd)
    Nmelt = 200 / max_topo
    
    # Initialize Wmax and scf with zeros
    Wmax = xr.zeros_like(swe) # threshold SWE above which SCF is 100%
    scf = xr.zeros_like(swe)
    
    
    # Iteratively update scf for each timestep
    for t in swe.time[1:]:
        # Current timestep's SWE and fsn
        swe_curr = swe.sel(time=t)
        
        # previous snow cover fraction
        fsn = scf.sel(time=t - np.timedelta64(1, "D")).fillna(0)
    
        # Delta SWE for subsequent timesteps
        delta_swe = swe_curr - swe.sel(time=t - np.timedelta64(1, "D"))
    
        # Accumulation
        is_accum = delta_swe > 0
        # s = np.minimum(1, kacc * delta_swe)
        s = np.tanh(kaccum * delta_swe)
        scf_accum = xr.where(is_accum, 
                           (1 - ((1 - s) * (1 - fsn))),
                           0)
        
        # Update Wmax for accumulation
        Wmax.loc[dict(time=t)] = xr.where(
            is_accum,
            np.minimum(swe_curr/(0.5*(np.cos(np.pi*(1-np.maximum(1e-6, fsn))**(1/Nmelt))+1)), 1e8),
            Wmax.sel(time=t - np.timedelta64(1, "D")).fillna(0)
        )
           
        # no snow condition
        no_snow = swe_curr==0
        
        # Reset Wmax for no snow condition
        Wmax.loc[dict(time=t)] = xr.where(
            no_snow,
            0,
            Wmax.loc[dict(time=t)]
        )
        
        # Melting
        is_melt = delta_swe < 0
    
        scf_melt = xr.where(
            is_melt,
            (1 - ((1 / np.pi) * np.arccos(2 * np.minimum(1, swe_curr / Wmax.sel(time=t)) - 1)) ** Nmelt),
            0
        )
        
        # Update scf for the current timestep
        scf.loc[dict(time=t)] = xr.where(swe_curr==0, 0, scf.loc[dict(time=t)])
        scf.loc[dict(time=t)] = xr.where(is_melt | is_accum, 
                                       scf_accum + scf_melt,
                                       scf.loc[dict(time=t- np.timedelta64(1, "D"))])
            
    scf = scf.fillna(0)
    scf = scf*100   
    
    return scf



def get_kaccum(scf, snow, array_ref):
    """
    Compute the parameter kacc used as input in the function "scf_param_swenson".
    It determines the first accumulation event, identifies the corresponding SCF 
    and computes kacc. The result is constrained to a maximum of 0.5, and 
    missing values are replaced with the default value 0.1.
    
    For details, see Swenson et Lawrence, 2012 
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JD018178


    Parameters:
    -----------
    scf : xarray.DataArray
        Snow Cover Fraction (SCF) in percentage (0-100)
    snow : xarray.DataArray
        Solid precipitation.
    array_ref : xarray.DataArray
        A reference DataArray to mask the output.

    Returns:
    --------
    kacc : xarray.DataArray
        Constant to be used in the parametrization.
    """
    
    # Determine the first accumulation
    accumulation = snow > 0
    
    # Index of the first accumulation (first day with snow observed by EO data)
    ix_first_acc = ((accumulation == 1) & (scf>0)).argmax(dim='time', skipna=True).compute()

    # Extract first SCF and first snow
    first_scf = scf.isel(time=ix_first_acc)
    first_snow = snow.isel(time=ix_first_acc)
    
    # Avoid division by zero in kacc calculation
    safe_snow = first_snow.where(first_snow > 0, np.nan)
    
    # Compute kacc
    kacc = np.arctanh(first_scf / 100) / safe_snow
    
    # Set upper limit
    kacc = xr.where(kacc > 0.5, 0.5, kacc)
    
    # Replace NaNs with 0.1
    kacc = kacc.fillna(0.1)
    
    #mask based on a reference array
    kacc = xr.where(array_ref.notnull(), kacc, np.nan)
    
    # Set variable name and attributes
    kacc.name = "kaccum"
    kacc.attrs = {
        "long_name": "kaccum",
        "units": "-",
        "description": "Constant calculated from the first accumulation.",
        "reference": "Swenson & Lawrence, 2012. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012JD018178"
    }
    
    return kacc







