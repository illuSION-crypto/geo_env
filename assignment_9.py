import xarray as xr
import os
import numpy as np
import geopandas as gpd
from scipy.stats import norm
from collections import namedtuple
import matplotlib.pyplot as plt
# Path to your directory
root_dir = r'/mnt/storage1/maj0d/data/ErSE316/assignment_9/'


# Create full paths
senerios = ['126', '370']
variables = ['Humidity', 'Temperature', 'Precipitation']
for s in senerios:
    for v in variables:
        output_path = os.path.join(root_dir, f'{v}{s}.nc')
        if os.path.exists(output_path):
            print(f"File {output_path} already exists, skipping...")
            continue
        subdir = os.path.join(root_dir, f'SSP{s}', f'{v}_{s}')
        full_paths = [os.path.join(subdir, f)
                      for f in os.listdir(subdir) if f.endswith('.nc')]
        # Open and concatenate along time dimension
        combined = xr.open_mfdataset(
            full_paths, combine='nested', concat_dim='time')
        # Save to new file
        combined.to_netcdf(output_path)
        print(f"Files successfully combined and saved to {output_path}")


# ----------------------------------------------------------
# TREND ANALYSIS FUNCTIONS (Hamed & Rao 1998 + Sen's Slope)
# ----------------------------------------------------------

def hamed_rao_mk_test(x, alpha=0.05):
    """Modified MK test with autocorrelation correction (Hamed & Rao 1998)"""
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])

    # Calculate variance with autocorrelation correction
    var_s = n*(n-1)*(2*n+5)/18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t*(t-1)*(2*t+5)/18

    # Correct for autocorrelation
    n_eff = n
    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, n//4)]
        n_eff = n / (1 + 2 * sum((n-i)/n * acf[i] for i in range(1, len(acf))))
        var_s *= n_eff / n

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1-alpha/2)

    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)


def sens_slope(x, y):
    """Sen's slope estimator"""
    slopes = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)


def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Calculate wet bulb temperature from air temperature and relative humidity. 
    Args:
        temp_k: Temperature in Kelvin
        rh_percent: Relative humidity in percent
    Returns:
       Wet bulb temperature in Kelvin
    """
    # Convert temperature from Kelvin to Celsius for calculations
    temp_c = temp_k - 273.15
    # Calculation using Stull's method (2011) - accurate to within 0.3°C
    wbt_c = temp_c * np.arctan(0.151977 * (rh_percent + 8.313659)**0.5) + \
        np.arctan(temp_c + rh_percent) - np.arctan(rh_percent - 1.676331) + \
        0.00391838 * (rh_percent)**(3/2) * \
        np.arctan(0.023101 * rh_percent) - 4.686035
    # Convert back to Kelvin
    wbt_k = wbt_c + 273.15
    return wbt_k


def main():
    fig_dir = r'/mnt/storage1/maj0d/projects/geo_env/figures/assignment_9/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    # Part 1: Load the datasets and plot annual averages
    for s in senerios:
        # Load the datasets
        temp_file = os.path.join(root_dir, f'Temperature{s}.nc')
        rh_file = os.path.join(root_dir, f'Humidity{s}.nc')
        prcp_file = os.path.join(root_dir, f'Precipitation{s}.nc')
        ds_temp = xr.open_dataset(temp_file).sortby('time').mean(dim=['lat', 'lon'])
        ds_rh = xr.open_dataset(rh_file).sortby('time').mean(dim=['lat', 'lon'])
        ds_prcp = xr.open_dataset(prcp_file).sortby('time').mean(dim=['lat', 'lon'])
        # Plot annual average
        annual_average_temp = ds_temp['tas'].groupby('time.year').mean(dim='time')
        annual_average_temp.plot()
        plt.title(f"Annual Average Temperature {s}")
        plt.xlabel('Year')
        plt.ylabel('Temperature (K)')
        plt.savefig(os.path.join(fig_dir, f'annual_avg_temp_{s}.png'))
        plt.close()
        annual_average_rh = ds_rh['hurs'].groupby('time.year').mean(dim='time')
        annual_average_rh.plot()
        plt.title(f"Annual Average Relative Humidity {s}")
        plt.xlabel('Year')
        plt.ylabel('Relative Humidity (%)')
        plt.savefig(os.path.join(fig_dir, f'annual_avg_rh_{s}.png'))
        plt.close()
        annual_average_prcp = ds_prcp['pr'].groupby('time.year').mean(dim='time')
        annual_average_prcp.plot()
        plt.title(f"Annual Average Precipitation {s}")
        plt.xlabel('Year')
        plt.ylabel('Precipitation (mm)')
        plt.savefig(os.path.join(fig_dir, f'annual_avg_prcp_{s}.png'))
        plt.close()
        # Trend analysis
        # Modified Mann-Kendall test
        adjusted_mk_test = hamed_rao_mk_test(
            annual_average_temp.values, alpha=0.05)
        print(f"Modified MK test for temperature {s}: {adjusted_mk_test}")
        adjusted_mk_test = hamed_rao_mk_test(
            annual_average_prcp.values, alpha=0.05)
        print(f"Modified MK test for precipitation {s}: {adjusted_mk_test}")
        sens_slope_temp = sens_slope(
            annual_average_temp['year'], annual_average_temp.values)
        print(f"Sen's slope for temperature {s}: {sens_slope_temp}")
        sens_slop_prcp = sens_slope(
            annual_average_prcp['year'], annual_average_prcp.values)
        print(f"Sen's slope for precipitation {s}: {sens_slop_prcp}")
        # Analysis of Climate Extremes
        # Take the annual average of all the grids for Saudi at daily steps
        annual_max_temp_data = ds_temp['tas'].resample(time='YE').max()
        annual_max_temp_data.plot()
        plt.title(f"Annual Maximum Temperature {s}")
        plt.xlabel('Date')
        plt.ylabel('Temperature (K)')
        plt.savefig(os.path.join(fig_dir, f'annual_max_temp_{s}.png'))
        plt.close()
        prcp_data = ds_prcp['pr'].resample(time='YE').max()
        prcp_data.plot()
        plt.title(f"Annual Maximum Precipitation {s}")
        plt.xlabel('Date')
        plt.ylabel('Precipitation (mm)')
        plt.savefig(os.path.join(fig_dir, f'annual_max_prcp_{s}.png'))
        plt.close()
        adjusted_mk_test = hamed_rao_mk_test(
            annual_max_temp_data.values, alpha=0.05)
        print(f"Modified MK test for annual max temperature {s}: {adjusted_mk_test}")
        adjusted_mk_test = hamed_rao_mk_test(
            prcp_data.values, alpha=0.05)
        print(f"Modified MK test for annual max precipitation {s}: {adjusted_mk_test}")
        # Output directory and file
        output_dir = root_dir
        output_file = os.path.join(output_dir, f"wb_{s}.nc")
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping...")
            wbt_k_ds = xr.open_dataset(output_file)
        else:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            # Extract temperature and humidity data
            temp_k = ds_temp['tas']  # Assuming 'tas' is temperature variable
            rh_percent = ds_rh['hurs']  # Assuming 'hurs' is relative humidity
            # Calculate wet bulb temperature
            wbt_k = calculate_wet_bulb_temperature(temp_k, rh_percent)
            # Create a new dataset for the output
            wbt_k_ds = xr.Dataset(
                {
                    'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_k.values),
                },
                coords={
                    'time': ds_temp['time'],
                    'lat': ds_temp['lat'],
                    'lon': ds_temp['lon'],
                },
                attrs={
                    'description': 'Wet bulb temperature calculated from temperature and relative humidity',
                    'units': 'K',
                    'calculation_method': "Stull's method (2011)",
                }
            )
            # Save to NetCDF
            wbt_k_ds.to_netcdf(output_file)
            print(f"Wet bulb temperature saved to: {output_file}")
        # Close the datasets
        ds_temp.close()
        ds_rh.close()
        # clip data to Saudi Arabia
        shapefile_path = '/mnt/storage1/maj0d/data/ErSE316/assignment_8/WS_3/WS_3.shp'
        gdf = gpd.read_file(shapefile_path)
        wbt_k_ds = wbt_k_ds.rio.write_crs("EPSG:4326")
        clipped_wbt_k = wbt_k_ds.rio.clip(gdf.geometry, gdf.crs, drop=True)['wet_bulb_temp']
        clipped_wbt_k = clipped_wbt_k.mean(dim=['lat', 'lon'])
        annual_average_wbt_k = clipped_wbt_k.resample(time='YE').mean()
        annual_average_wbt_k.plot()
        plt.title(f"Annual Average Wet Bulb Temperature {s}")
        plt.xlabel('Year')
        plt.ylabel('Wet Bulb Temperature (K)')
        plt.savefig(os.path.join(fig_dir, f'annual_avg_wbt_{s}.png'))
        plt.close()
        # Trend analysis for wet bulb temperature
        adjusted_mk_test = hamed_rao_mk_test(
            annual_average_wbt_k.values, alpha=0.05)
        print(f"Modified MK test for wet bulb temperature {s}: {adjusted_mk_test}")
        sens_slope_wbt = sens_slope(
            annual_average_wbt_k['time.year'].values, annual_average_wbt_k.values)
        print(f"Sen's slope for wet bulb temperature {s}: {sens_slope_wbt}")
        annual_max_wbt_k = clipped_wbt_k.resample(time='YE').max()
        annual_max_wbt_k.plot()
        plt.title(f"Annual Maximum Wet Bulb Temperature {s}")
        plt.xlabel('Date')
        plt.ylabel('Wet Bulb Temperature (K)')
        plt.savefig(os.path.join(fig_dir, f'annual_max_wbt_{s}.png'))
        plt.close()
        adjusted_mk_test = hamed_rao_mk_test(
            annual_max_wbt_k.values, alpha=0.05)
        print(f"Modified MK test for annual max wet bulb temperature {s}: {adjusted_mk_test}")
        sens_slope_wbt = sens_slope(
            annual_max_wbt_k['time.year'].values, annual_max_wbt_k.values)
        print(f"Sen's slope for annual max wet bulb temperature {s}: {sens_slope_wbt}")
        wbt_k_ds.close()

if __name__ == "__main__":
    main()
