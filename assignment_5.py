import os
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
# Data Visualization
data_folder = r'/mnt/storage1/maj0d/data/ErSE316/assignment_5'
fig_dir= '/mnt/storage1/maj0d/projects/geo_env/figures/assignment_5'
dset = xr.open_dataset(os.path.join(data_folder, 'GRIDSAT-B1.2009.11.25.12.v02r01.nc'))
IR = np.array(dset.variables['irwin_cdr']).squeeze()
IR = np.flipud(IR)
IR = IR*0.01+200
IR = IR-273.15
plt.figure(1)
plt.imshow(IR, extent=[-180.035, 180.035, -70.035, 70.035], aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')

jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker='o', label='Jeddah')
plt.title('Brightness temperature at 12:00 UTC on 2009.11.25')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.savefig(os.path.join(fig_dir, 'IR.png'))

#  Rainfall Estimation
# Calculate the cumulative rainfall between 00:00 and 12:00 UTC
data_date = '2009.11.25'
freq = 3
start_time = 0
end_time = 12
time_range = range(start_time,end_time+1,freq)
A = 1.1183*(10**11) # mm/h
b = 3.6382*(10**-2) # K^-1
print(b)
c = 1.2 # empirical constant
jeddah_rainfalls = []
brightness_temps = []
for t in time_range:
    dset = xr.open_dataset(os.path.join(data_folder, f'GRIDSAT-B1.{data_date}.{t:02d}.v02r01.nc'))
    jeddah_ir = dset.sel(lon=jeddah_lon, lat=jeddah_lat, method='nearest').irwin_cdr
    jeddah_ir = np.array(jeddah_ir,dtype=np.float64).squeeze()
    brightness_temp = jeddah_ir*0.01+200
    brightness_temps.append(brightness_temp)
    rainfall_rate = A * np.exp(-b*np.power(brightness_temp, c))
    print(f'Rainfall rate at {t:02d}:00 UTC: {rainfall_rate} mm/h')
    jeddah_rainfalls.append(rainfall_rate)
plt.figure(2)
plt.plot(time_range, jeddah_rainfalls)
plt.xlabel('Time (UTC)')
plt.ylabel('Rainfall rate (mm/h)')
plt.title('Rainfall rate at Jeddah between 00:00 and 12:00 UTC')
plt.savefig(os.path.join(fig_dir, 'rainfall_rate.png'))
plt.figure(3)
plt.plot(time_range, brightness_temps)
plt.xlabel('Time (UTC)')
plt.ylabel('Brightness temperature (degrees Celsius)')
plt.title('Brightness temperature at Jeddah between 00:00 and 12:00 UTC\n')
plt.savefig(os.path.join(fig_dir, 'brightness_temp.png'))