import tools
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_path = '/mnt/storage1/maj0d/data/ErSE316/assignment_6/era5.nc'
figure_dir = '/mnt/storage1/maj0d/projects/geo_env/figures/assignment_6'
dset = xr.open_dataset(data_path)
t2m = np.array(dset['t2m'])
tp = np.array(dset['tp'])
lat = np.array(dset['latitude'])
lon = np.array(dset['longitude'])
time = np.array(dset['valid_time'])
t2m = t2m-273.15
tp = tp*1000
if t2m.ndim == 4:
    t2m = np.nanmean(t2m, axis=1)
    tp = np.nanmean(tp, axis=1)

df_era5 = pd.DataFrame(index=time)
df_era5['t2m'] = t2m[:,3,2]
df_era5['tp'] = tp[:,3,2]
df_era5.plot()
plt.title('Time Series of Temperature and Precipitation')
plt.xlabel('Time')
plt.ylabel('Temperature (C)\n Precipitation (mm/day)')
plt.savefig(os.path.join(figure_dir, 'time_series.png'))
plt.clf()
annual_precip = df_era5['tp'].resample('YE').mean()*24*365.25
mean_annual_precipitation = np.nanmean(annual_precip)
print(f'Mean Annual Precipitation: {mean_annual_precipitation} mm/year')
# Calculation of Potential Evaporation
tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 21.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear

pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)
ts_index = df_era5['t2m'].resample('D').mean().index
plt.figure()
plt.plot(ts_index, pe,label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential Evaporation (mm/day)')
plt.title('Potential Evaporation')
plt.savefig(os.path.join(figure_dir, 'potential_evaporation.png'))

df_pe = pd.DataFrame(index=df_era5['t2m'].resample('D').mean().index)
df_pe['pe'] = pe
annual_pe = df_pe['pe'].resample('YE').mean()
mean_annual_pe = np.nanmean(annual_pe)
print(f'Mean Annual Potential Evaporation: {mean_annual_pe} mm/year')
area = 1.6e6 # m^2
volume_evaporation = mean_annual_pe*area*1e-3
print(f'Volume of Evaporation: {volume_evaporation} m^3/year')

