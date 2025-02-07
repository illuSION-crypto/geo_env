import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
from scipy import stats

#  Part 2: Exploring the Data
dir_path = "/mnt/storage1/maj0d/projects/data/ErSE316/assignment_2"
figures_dir = "/mnt/storage1/maj0d/projects/geo_env/figures/assignment_2"
fnames = os.listdir(dir_path)
for fname in fnames:
    print("the name of the netCDF file is:", fname)
    data_path = os.path.join(dir_path, fname)
    dset = xr.open_dataset(data_path)
    # check the variables
    print("this dataset has following variables:")
    print(dset.data_vars)
    # check the dimensions of air temperature variable
    air_temperature = dset['tas']
    print("the dimension of temperature is:")
    print(air_temperature.dims)
    for dim in air_temperature.dims:
        print(f"the name of dimension: {dim}, the length of it: {air_temperature.sizes[dim]}")
    # type of data value
    print("the type of data value is:", air_temperature.dtype)
    # time span
    print("the time span is:",dset.coords['time'].values[0], dset.coords['time'].values[-1])
    # the unit of temperature
    print("the unit of temperature is:", air_temperature.units)
    # spatial and temporal resolution
    for coord in ['lat', 'lon', 'time']:
        resolution = air_temperature.coords[coord].diff(coord)[0].item()
        if coord == 'time':
            # convert the resolution from nanosecond to days
            resolution = resolution/1e9/3600/24
        print(f"the resolution of {coord} is: {'%.2f'%resolution}")
# Part 3: Creation of Climate Change Maps
 # Calculate the mean air temperature map for 1850–1900
air_temperature = xr.open_dataset(os.path.join(dir_path, 'tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc'))['tas']
mean_1850_1900 = air_temperature.sel(time=slice('1850-01-01', '1900-12-31')).mean(dim='time')
print("The properties of mean temperature:")
print(f"dtype,{mean_1850_1900.dtype},shape:{mean_1850_1900.shape}")
img = mean_1850_1900.plot()
cbar = img.colorbar
cbar.set_label('Temperature (K)')
time_range = f"1850-1900"
plt.title(f'{time_range} Mean Air Temperature')
plt.savefig(f'{time_range}_mean_temperature.png')
plt.clf()
# Calculate the mean air temperature map for 2071-2100 for each scenario
scenarios = ['ssp119', 'ssp245', 'ssp585']
yearly_series = []
for scenario in scenarios:
    air_temperature = xr.open_dataset(os.path.join(dir_path, f'tas_Amon_GFDL-ESM4_{scenario}_r1i1p1f1_gr1_201501-210012.nc'))['tas']
    mean_2071_2100 = air_temperature.sel(time=slice('2071-01-01', '2100-12-31')).mean(dim='time')
    img = mean_2071_2100.plot()
    cbar = img.colorbar
    cbar.set_label('Temperature (K)')
    time_range = f"2071-2100"
    plt.title(f'{scenario} {time_range} Mean Air Temperature')
    fig_name = f'{scenario}_{time_range}_mean_temperature.png'
    fig_path = os.path.join(figures_dir, fig_name)
    plt.savefig(fig_path,dpi=300)
    plt.clf()
    # Calculate the difference between the mean air temperature map for 2071-2100 and 1850-1900
    diff = mean_2071_2100 - mean_1850_1900
    img = diff.plot()
    cbar = img.colorbar
    cbar.set_label('Temperature (K)')
    time_range = f"2071-2100"
    plt.title(f'{scenario} Mean Air Temperature Difference\nbetween {time_range} and 1850-1900')
    fig_name = f'{scenario}_{time_range}_mean_temperature_diff.png'
    fig_path = os.path.join(figures_dir, fig_name)
    plt.savefig(fig_path,dpi=300)
    plt.clf()
    # Calculate air temperature trend
    # decade_mean = air_temperature.mean(dim='lon').mean(dim='lat').resample(time='10YE').mean()
    # decade_mean.plot()
    # plt.title(f'{scenario} Air Temperature Trend')
    # plt.savefig(f'{scenario}_temperature_trend.png')
    # plt.clf()
    
    # print(f"{scenario} air temperature trend: {slope:.4f} K/decade")
    annual_series = air_temperature.mean(dim='lon').mean(dim='lat').resample(time='YE').mean()
    years = annual_series.time.dt.year
    yearly_series.append({
        "scenario": scenario,
        "years": years,
        "annual_series": annual_series
    })
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(years, annual_series)
    print(f"{scenario} air temperature trend: {(slope*10):.4f} K/decade")

for series in yearly_series:
    plt.plot(series["years"], series["annual_series"], label=series["scenario"])
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.title('Annual Mean Air Temperature')
plt.legend()
fig_name = 'annual_mean_temperature.png'
fig_path = os.path.join(figures_dir, fig_name)
plt.savefig(fig_path,dpi=300)
    
    
    
    
    


    

    
    