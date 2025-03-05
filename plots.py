#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:10:15 2025

@author: vpremier
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter



def plot_snowmelt_coeff(basin, cm_l, cm_eo1_mean, cm_eo2_mean):
    """
    Plots snowmelt coefficients for a given basin, including maps and histograms.
    
    This function generates a figure with three maps and corresponding histograms,
    showing snowmelt coefficients from different datasets. The maps use a consistent 
    color scale for comparison, and a shared color bar is added for reference.
    
    Parameters:
    basin (str): Name of the basin.
    cm_l (xarray.DataArray): LISFLOOD snowmelt coefficient.
    cm_eo1_mean (xarray.DataArray): Snowmelt coefficient from Pistocchi et al. 2017.
    cm_eo2_mean (xarray.DataArray): Snowmelt coefficient from optimization. 
    
    Returns:
    None: Displays the plots.
    """
    # Dictionary with correct names
    nameList = {'Adige' : 'Adige',
                  'Alpenrhein' : 'Alpenrhein',
                  'Arve' : 'Arve',
                  'Gallego' : 'Gállego',
                  'Guadalfeo' : 'Guadalfeo',
                  'Laborec' : 'Laborec',
                  'Morrumsan' : 'Mörrumsån',
                  'Salzach' : 'Salzach',
                  'Umealven' : 'Umeälven'}
    
    # Create a GridSpec layout with 1 row and 6 columns
    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[3, 1, 3,1, 3, 1])  # Allocate space for 3 plots and 3 histograms
    
    # Define vmin and vmax
    vmin = 0.5
    vmax = 10
    
    # Set the tick positions (shifted inward)
    tick_shift = 0.2  # Fraction of the axis length for inward shift
    
    # First map and histogram
    ax1 = fig.add_subplot(gs[0, 0])  # First map
    cm_l.plot(ax=ax1, cmap='magma', vmin=vmin, vmax=vmax, add_colorbar=False)
    ax1.set_title(f'L-C$_{{m}}$', fontsize=16)
    ax1.set_xlabel("", fontsize=14)
    ax1.set_ylabel("", fontsize=14)
    
    # Add a grid to the first map
    ax1.grid(True, linestyle='--', color='gray', linewidth=0.5)
    
    # Calculate and set custom ticks for the first map
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    xticks = [xlim[0] + tick_shift * (xlim[1] - xlim[0]), xlim[1] - tick_shift * (xlim[1] - xlim[0])]
    yticks = [ylim[0] + tick_shift * (ylim[1] - ylim[0]), ylim[1] - tick_shift * (ylim[1] - ylim[0])]
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)
    
    # Rotate y-axis tick labels
    ax1.tick_params(axis='y', labelrotation=90, labelsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    
    # Format tick labels
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax1_hist = fig.add_subplot(gs[0, 1])  # First histogram
    ax1_hist.hist(cm_l.values.flatten(), bins=30, color='gray', range=(vmin, vmax))
    ax1_hist.set_ylabel("Frequency", fontsize=14)
    ax1_hist.set_xlim([0,10])
    
    # Format the histogram y-axis in scientific notation
    ax1_hist.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1_hist.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1_hist.tick_params(labelsize=14)
    
    # Third map and histogram
    ax2 = fig.add_subplot(gs[0, 2])  # Third map
    cbar_eo1 = cm_eo1_mean.plot(ax=ax2, cmap='magma', vmin=vmin, vmax=vmax, add_colorbar=False)
    ax2.set_title(f'EO-C$_{{m,1}}$', fontsize=16)
    ax2.set_xlabel("", fontsize=14)
    ax2.set_ylabel("", fontsize=14)
    
    # Add a grid to the second map
    ax2.grid(True, linestyle='--', color='gray', linewidth=0.5)
    
    # Calculate and set custom ticks for the second map
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    xticks = [xlim[0] + tick_shift * (xlim[1] - xlim[0]), xlim[1] - tick_shift * (xlim[1] - xlim[0])]
    yticks = [ylim[0] + tick_shift * (ylim[1] - ylim[0]), ylim[1] - tick_shift * (ylim[1] - ylim[0])]
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    
    # Rotate y-axis tick labels
    ax2.tick_params(axis='y', labelrotation=90, labelsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    
    # Format tick labels
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax2_hist = fig.add_subplot(gs[0, 3])  # Third histogram
    ax2_hist.hist(cm_eo1_mean.values.flatten(), bins=30, color='gray', range=(vmin, vmax))
    ax2_hist.set_ylabel("Frequency", fontsize=14)
    ax2_hist.set_xlim([0,10])
    
    # Format the histogram y-axis in scientific notation
    ax2_hist.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2_hist.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2_hist.tick_params( labelsize=14)
    
    # Second map and histogram
    ax3 = fig.add_subplot(gs[0, 4])  # Second map
    cm_eo2_mean.plot(ax=ax3, cmap='magma', vmin=vmin, vmax=vmax, add_colorbar=False)
    ax3.set_title(f'EO-C$_{{m,2}}$', fontsize=16)
    ax3.set_xlabel("", fontsize=14)
    ax3.set_ylabel("", fontsize=14)
    
    # Add a grid to the third map
    ax3.grid(True, linestyle='--', color='gray', linewidth=0.5)
    
    # Calculate and set custom ticks for the third map
    xlim = ax3.get_xlim()
    ylim = ax3.get_ylim()
    xticks = [xlim[0] + tick_shift * (xlim[1] - xlim[0]), xlim[1] - tick_shift * (xlim[1] - xlim[0])]
    yticks = [ylim[0] + tick_shift * (ylim[1] - ylim[0]), ylim[1] - tick_shift * (ylim[1] - ylim[0])]
    ax3.set_xticks(xticks)
    ax3.set_yticks(yticks)
    
    # Rotate y-axis tick labels
    ax3.tick_params(axis='y', labelrotation=90, labelsize=14)
    ax3.tick_params(axis='x', labelsize=14)
    
    # Format tick labels
    ax3.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax3_hist = fig.add_subplot(gs[0, 5])  # Second histogram
    ax3_hist.hist(cm_eo2_mean.values.flatten(), bins=30, color='gray', range=(vmin, vmax))
    ax3_hist.set_ylabel("Frequency", fontsize=14)
    ax3_hist.set_xlim([0,10])
    
    # Format the histogram y-axis in scientific notation
    ax3_hist.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax3_hist.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax3_hist.tick_params(labelsize=14)
    
    
    # Add a shared colorbar
    
    # cbar = fig.colorbar(cbar_eo1, ax=[ax1, ax2, ax3], orientation='horizontal', fraction=0.05, pad=0.1)
    # cbar.ax.set_position([0.5, 0.94, 0.7, 0.045])  # Adjust position to be near the title
    # Add a general title to the entire plot
    fig.suptitle(nameList[basin], fontsize=18)
    
    # Adjust layout
    plt.tight_layout()
    
    
    
def get_sca(scf, mask):
    """
    Computes the mean snow cover area (SCA).
    
    Parameters:
    -----------
    scf : xarray.DataArray
        SF time-series.
    mask : numpy.ndarray
        Mask with the area of interest.
    
    Returns:
    --------
    xarray.DataArray
        A DataArray representing the mean value of `sca` over the valid (non-masked) pixels 
        for each time step.
    """


    valid_pixels = scf.sizes['lon']*scf.sizes['lat']-np.nansum(mask)
    
    mask_da = xr.DataArray(mask, coords=[scf.lat, scf.lon], dims=["lat", "lon"])
    mask_broadcasted = mask_da.expand_dims(time=scf.time)
    
    sca = (scf.where(~mask_broadcasted)).sum(dim=['lon', 'lat'])/valid_pixels
    
    return sca



def plot_sca(basin, scf, scf_l, scf_eo1, scf_eo2, mask):
    """
    Plots the Snow Cover Area (SCA) over time for the EO dataset and LISFLOOD 
    withe the different snowmelt coefficient.

    Parameters:
    -----------
    basin : str
        The name of the river basin being analyzed. Used for labeling the plot.
    
    scf : xarray.DataArray
        The benchmark Snow Cover Fraction (SCF) dataset (EO-SCA) over time.
    
    scf_l : xarray.DataArray
        The SCF computed using the LISFLOOD model with the standard snowmelt coefficient.
    
    scf_eo1 : xarray.DataArray
        The SCF computed using LISFLOOD with an EO-based snowmelt coefficient from Pistocchi et al., 2017.
    
    scf_eo2 : xarray.DataArray
        The SCF computed using LISFLOOD with an EO-based optimized snowmelt coefficient.
    
    cm_l : xarray.DataArray
        The LISFLOOD snowmelt coefficient used as mask.
    
    Returns:
    --------
    None
        The function generates and displays a time series plot of SCA for the different models.

    """

    
    # Dictionary with correct names
    nameList = {'Adige' : 'Adige',
                  'Alpenrhein' : 'Alpenrhein',
                  'Arve' : 'Arve',
                  'Gallego' : 'Gállego',
                  'Guadalfeo' : 'Guadalfeo',
                  'Laborec' : 'Laborec',
                  'Morrumsan' : 'Mörrumsån',
                  'Salzach' : 'Salzach',
                  'Umealven' : 'Umeälven'}
    
    colors = ['black', '#56B4E9', '#D55E00', '#009E73']  # These are colorblind-friendly colors (from the CUD palette)

    
    # get the snow cover area (SCA) over the masked area
    sca_eo = get_sca(scf, mask) # EO benchmark
    sca_l = get_sca(scf_l, mask) # LISFLOOD with standard snowmelt coeff
    sca_eo_cm1 = get_sca(scf_eo1, mask) # LISFLOOD with EO snowmelt coeff with Pistocchi et al., 2017
    sca_eo_cm2 = get_sca(scf_eo2, mask) # LISFLOOD with EO snowmelt coeff with optimization
    
    

    # create the figure 
    plt.figure(figsize=(24, 3))
    plt.plot(scf.time, sca_eo.values, label='EO-SCA', color=colors[0], linewidth=1, linestyle='-')
    plt.plot(scf.time, sca_l.values, label ='L-SCA L-C$_{m}$', color=colors[1], linewidth=1, linestyle='-')
    plt.plot(scf.time, sca_eo_cm1.values, label ='L-SCA EO-C$_{m,1}$', color=colors[3], linewidth=1, linestyle='-')
    plt.plot(scf.time, sca_eo_cm2.values, label ='L-SCA EO-C$_{m,2}$', color=colors[2], linewidth=1, linestyle='-')


    plt.ylim(-1, 101)
    plt.ylabel("SCA [%]")

    # Add grid lines for clarity
    plt.grid(alpha=0.3)
    
    # Hide x-tick labels
    # if basin != 'Umealven':
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.text(.45, .95, nameList[basin], ha='left', va='top', transform=ax.transAxes, fontsize=22)
    plt.xlim(scf.time[0], scf.time[-1])

    # Enhance layout
    plt.tight_layout()
      
    

def compute_pixelwise_statistics(modelled, target, mask):
    """
    Computes pixel-wise statistics (bias, RMSE, and correlation) over time.
    Only SCF values where either the modelled or target data differ from 0 
    are considered for the computation.

    Parameters:
    -----------
    modelled : xarray.DataArray
        The modelled data (e.g., SCF from LISFLOOD).
    
    target : xarray.DataArray
        The reference/target data (e.g., observed SCF).
    
    mask : numpy.ndarray
        A 2D mask array (lat, lon) where `1` indicates masked (invalid) pixels, 
        and `0` indicates valid pixels.
    
    Returns:
    --------
    avg_bias : xarray.DataArray
        Time-averaged pixel-wise bias (modelled - target), with dimensions (lat, lon).
    
    avg_rmse : xarray.DataArray
        Time-averaged pixel-wise Root Mean Square Error (RMSE), with dimensions (lat, lon).
    
    correlation : xarray.DataArray
        Pixel-wise correlation coefficient between modelled and target data over time, with dimensions (lat, lon).
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
    avg_bias = pixel_bias.mean(dim="time")  # Average bias over time for each pixel
    
    # Calculate pixel-wise RMSE
    pixel_squared_error = ((modelled - target) ** 2).where(valid_condition)
    avg_rmse = np.sqrt(pixel_squared_error.mean(dim="time"))  # Time-averaged RMSE

    # Calculate pixel-wise correlation
    modelled_mean = modelled.where(valid_condition).mean(dim="time")
    target_mean = target.where(valid_condition).mean(dim="time")
    modelled_anomaly = modelled - modelled_mean
    target_anomaly = target - target_mean

    numerator = (modelled_anomaly * target_anomaly).where(valid_condition).sum(dim="time")
    denominator = np.sqrt(
        (modelled_anomaly ** 2).where(valid_condition).sum(dim="time") *
        (target_anomaly ** 2).where(valid_condition).sum(dim="time")
    )
    correlation = numerator / denominator

    return avg_bias, avg_rmse, correlation



def print_metrics(modelled, target, mask):
    
    bias, rmse, corr = compute_pixelwise_statistics(modelled, target, mask)  
    
    print('BIAS = %.2f' % np.nanmean(bias.values))
    print('RMSE = %.2f' % np.nanmean(rmse.values))
    print('corr = %.2f' % np.nanmean(corr.values))
    
