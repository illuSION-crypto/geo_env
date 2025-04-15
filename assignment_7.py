import tools
import os
import rioxarray
from shapely.geometry import mapping
import geopandas as gpd
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
# Part 2. Data Visualization
figure_dir = '/mnt/storage1/maj0d/projects/geo_env/figures/assignment_7'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
root_dir = '/mnt/storage1/maj0d/data/ErSE316/assignment_7/ERA5_data'
var_names = os.listdir(root_dir)
var_dict = {}
shp_path = '/mnt/storage1/maj0d/data/ErSE316/assignment_7/Saudi_Shape_File/Saudi_Shape.shp'
gdf = gpd.read_file(shp_path)
gdf = gdf.to_crs('EPSG:4326')
for var_name in var_names:
    dset = xr.open_mfdataset(
        os.path.join(root_dir, var_name, '*.nc'))
    print('This dataset has following variables:',list(dset.keys()))
    data_var = dset[list(dset.keys())[0]]
    if var_name == 'Total_Evaporation':
        data_var.values = data_var.values * (-1)
    data_var = data_var.rio.set_spatial_dims('longitude', 'latitude')
    data_var = data_var.rio.write_crs('EPSG:4326')
    unit = data_var.units
    data_var_clipped = data_var.rio.clip(gdf.geometry.apply(mapping), drop=True)
    monthly_sum =  data_var_clipped.resample(valid_time='1ME').sum('valid_time').mean(dim=('latitude', 'longitude'))*1000
    yearly_sum = monthly_sum.resample(valid_time='1YE').sum('valid_time')
    # yearly_sum['valid_time'] = yearly_sum['valid_time'].to_pandas()-pd.DateOffset(months=12)+pd.DateOffset(days=1)
    # monthly_sum['valid_time'] = monthly_sum['valid_time'].to_pandas()-pd.DateOffset(months=1)+pd.DateOffset(days=1)
    plt.figure(figsize=(15,7))
    yearly_sum.plot(linestyle='--',color='red',marker='o',label='Yearly Sum')
    monthly_sum.plot(color='blue',label='Monthly Sum')
    plt.title(f'Monthly/Yearly {var_name} Sum in Saudi Arabia')
    plt.xlabel('Year')
    plt.ylabel(f'{var_name} ({unit})')
    plt.legend()
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每年一个刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # 格式化为4位年份
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(figure_dir, f'monthly_{var_name.lower()}_sum.png'))
    plt.clf()
    var_dict[var_name] = monthly_sum
    dset.close()
difference = var_dict['Precipitation'] - var_dict['Total_Evaporation'] - var_dict['Runoff']
difference.plot(color='blue',label='Monthly')
yearly_sum = difference.resample(valid_time='1YE').sum('valid_time')
# yearly_sum['valid_time'] = yearly_sum['valid_time'].to_pandas()-pd.DateOffset(months=12)+pd.DateOffset(days=1)
yearly_sum.plot(linestyle='--',color='red',marker='o',label='Yearly')
plt.title('Difference between Precipitation and Evaporation plus Runoff in Saudi Arabia')
plt.xlabel('Year')
plt.ylabel('Difference (m)')
plt.grid()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每年一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # 格式化为4位年份
plt.xticks(rotation=45)
plt.savefig(os.path.join(figure_dir, 'difference_precipitation_evaporation_runoff.png'))
plt.clf()
# Compare runoff with precipitation - evaporation
runoff = var_dict['Runoff']
runoff.plot(color='blue',label='Monthly Runoff')
precipitation_evaporation = var_dict['Precipitation'] - var_dict['Total_Evaporation']
precipitation_evaporation.plot(color='blue',linestyle='--',label='Monthly Precipitation - Evaporation')
yearly_runoff = runoff.resample(valid_time='1YE').sum('valid_time')
# yearly_runoff['valid_time'] = yearly_runoff['valid_time'].to_pandas()-pd.DateOffset(months=12)+pd.DateOffset(days=1)
yearly_precipitation_evaporation = precipitation_evaporation.resample(valid_time='1YE').sum('valid_time')
# yearly_precipitation_evaporation['valid_time'] = yearly_precipitation_evaporation['valid_time'].to_pandas()-pd.DateOffset(months=12)+pd.DateOffset(days=1)
yearly_runoff.plot(color='red',marker='o',label='Yearly Runoff')
yearly_precipitation_evaporation.plot(linestyle='--',color='red',marker='o',label='Yearly Precipitation - Evaporation')
plt.title('Runoff and Precipitation - Evaporation in Saudi Arabia')
plt.xlabel('Year')
plt.ylabel('Runoff/Precipitation - Evaporation (m)')
plt.grid()
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator(1))  # 每年一个刻度
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  # 格式化为4位年份
plt.xticks(rotation=45)
plt.legend()
plt.savefig(os.path.join(figure_dir, 'runoff_precipitation_evaporation.png'))
plt.clf()
# calculate correlation between runoff and precipitation
runoff = var_dict['Runoff']
precipitation = var_dict['Precipitation']
correlation = np.corrcoef(runoff,precipitation)
print('Correlation between runoff and precipitation:',correlation[0,1])




    


