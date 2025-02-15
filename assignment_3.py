import tools
import os
import xarray as xr
from matplotlib import pyplot as plt

# Part 2. Heat Index Calculation
fig_dir = '/mnt/storage1/maj0d/projects/geo_env/figures/assignment_3'
data_path = '/mnt/storage1/maj0d/projects/data/ErSE316/assignment_3/41024099999.csv'
# load data
df_isd = tools.read_isd_csv(data_path)
plot = df_isd.plot(title="ISD data for Jeddah")
plt.savefig(os.path.join(fig_dir, 'isd_data.png'))
plt.clf()
# Calculate Heat Index (HI)
df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values,df_isd['TMP'].values)
df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)
highest_values = df_isd.max()
highest_hi = highest_values['HI']
highest_hi_F = highest_hi * 9/5 + 32
print('Highest HI is {} (Celsius), {}, (Fahrenheit)'.format(highest_hi, highest_hi_F))
highest_idx = df_isd.idxmax()
print('Highest HI is at {} (UTC)'.format(highest_idx['HI']))
# Convert to local time
local_time = highest_idx['HI'].tz_localize('UTC').tz_convert('Asia/Riyadh')
print('Highest HI is at {} (local time)'.format(local_time))
print('Data at highest HI:')
print(df_isd.loc[highest_idx['HI']])
# Check the possibility of heatwave
month_highest = highest_idx['HI'].month
year_highest = highest_idx['HI'].year
data_month = df_isd.loc[f'{year_highest}-{str(month_highest).zfill(2)}']
data_month['HI'].plot(title="Heat Index for the month of highest HI")
plt.xlabel('Date')
plt.ylabel('Heat Index (Celsius)')
plt.savefig(os.path.join(fig_dir, 'HI_month.png'))
plt.clf()
# Calculate HI based on daily means
df_isd['HI'].plot(title="Heat Index for Jeddah",label='Hourly')
daily_means = df_isd.resample('D').mean()
daily_means['RH'] = tools.dewpoint_to_rh(daily_means['DEW'].values, daily_means['TMP'].values)
daily_means['HI'] = tools.gen_heat_index(daily_means['TMP'].values, daily_means['RH'].values)
daily_means['HI'].plot(label='Daily')
plt.xlabel('Date')
plt.ylabel('Heat Index (Celsius)')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'HI.png'))
plt.clf()
# Part 3. Potential Impact of Climate Change
# Load data
global_ssp245_path = '/mnt/storage1/maj0d/projects/data/ErSE316/assignment_2/tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc'
global_historical_path = '/mnt/storage1/maj0d/projects/data/ErSE316/assignment_2/tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc'
global_ssp245 = xr.open_dataset(global_ssp245_path)
global_historical = xr.open_dataset(global_historical_path)
# Calculate Projected Increase in Temperature for Jeddah
# start_time = df_isd.index[0].to_pydatetime().date().strftime('%Y-%m-%d')
# end_time = df_isd.index[-1].to_pydatetime().date().strftime('%Y-%m-%d')
jeddah_coords_isd = {'lat': 21.679564, 'lon': 39.156536}
jeddah_historical = global_historical['tas'].sel(lat=jeddah_coords_isd['lat'], lon=jeddah_coords_isd['lon'],method='nearest')
jeddah_projection = global_ssp245['tas'].sel(lat=jeddah_coords_isd['lat'], lon=jeddah_coords_isd['lon'],method='nearest')
jeddah_coords_globe = jeddah_historical.coords
lat_error = abs((jeddah_coords_globe['lat'] - jeddah_coords_isd['lat']).values.item())
lon_error = abs((jeddah_coords_globe['lon'] - jeddah_coords_isd['lon']).values.item())
print('The nearest latitude and longitude for Jeddah on the globe are: ({:.3f}, {:.3f}). The coordinates of it in isd data are ({:.3f},{:.3f}). The coordinates error are {:.3f},{:.3f}'.format(jeddah_coords_globe['lat'].values, jeddah_coords_globe['lon'].values, jeddah_coords_isd['lat'], jeddah_coords_isd['lon'], lat_error, lon_error))
mean_1850_1900 = jeddah_historical.sel(time=slice('1850-01-01', '1900-12-31')).mean()
mean_2071_2100 = jeddah_projection.sel(time=slice('2071-01-01', '2100-12-31')).mean()
projected_increase = (mean_2071_2100 - mean_1850_1900).values.item()
print(f'Projected increase in temperature for Jeddah: {projected_increase:.3f} Celsius')
# apply the increase to the ISD data
df_isd['TMP_proj'] = df_isd['TMP'] + projected_increase
df_isd['RH_proj'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP_proj'].values)
df_isd['HI_proj'] = tools.gen_heat_index(df_isd['TMP_proj'].values, df_isd['RH_proj'].values)
df_isd['HI'].plot(title="Heat Index with climate change",label='Observed',alpha=0.5)
df_isd['HI_proj'].plot(label='Projected',alpha=0.5)
plt.legend(['Observed', 'Projected'])
plt.xlabel('Date')
plt.ylabel('Heat Index (Celsius)')
plt.savefig(os.path.join(fig_dir, 'HI_projected_increase.png'))
plt.clf()
# Calculate the difference
difference_HI = df_isd['HI_proj'] - df_isd['HI']
difference_HI.plot(title="Difference in Heat Index with climate change")
plt.xlabel('Date')
plt.ylabel('Difference (Celsius)')
plt.savefig(os.path.join(fig_dir, 'HI_difference.png'))
highest_hi_proj = df_isd['HI_proj'].max()
highest_idx_proj = df_isd['HI_proj'].idxmax()
print('Highest HI with climate change is {:.3f} (Celsius)'.format(highest_hi_proj))
print('Highest HI with climate change is at {} (UTC)'.format(highest_idx_proj))
local_time_proj = highest_idx_proj.tz_localize('UTC').tz_convert('Asia/Riyadh')
print('Highest HI with climate change is at {} (local time)'.format(local_time_proj))
difference_between = highest_hi_proj - highest_hi
print('Difference in highest HI with climate change: {:.3f} Celsius'.format(difference_between))