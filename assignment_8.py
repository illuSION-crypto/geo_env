from pyparsing import line
import xarray as xr
import geopandas as gpd
import numpy as np
import scipy.optimize as opt
import os
import matplotlib.pyplot as plt

## ---Part 1: Pre-Processing ERA5 dataset ---
# Clip each variable using the shapefile
def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    ds = ds.rio.write_crs("EPSG:4326")  # Ensure correct CRS
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped[var_name]

data_dir = '/mnt/storage1/maj0d/data/ErSE316/assignment_8'
fig_dir = '/mnt/storage1/maj0d/projects/geo_env/figures/assignment_8'
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
# Load the watershed shapefile
shapefile_path = os.path.join(data_dir,"WS_3/WS_3.shp")
gdf = gpd.read_file(shapefile_path)

# Load the NetCDF files (precipitation, ET, runoff)
precip_file = os.path.join(data_dir,"era5_OLR_2001_total_precipitation.nc")
et_file = os.path.join(data_dir,"era5_OLR_2001_total_evaporation.nc")
runoff_file = os.path.join(data_dir,"ambientera5_OLR_2001_total_runoff.nc")
# Extract variables
# Load and clip each dataset,unit conversion: meters to mm
P_grid = load_and_clip(precip_file, "tp", gdf) * 1000.0
ET_grid = load_and_clip(et_file, "e", gdf) * 1000.0
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000.0

# Compute area-averaged values
P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET = np.where(ET < 0.0, -ET, ET) 
# Plotting
plt.plot(P, label='Precipitation (mm)', color='blue',alpha=0.7)
plt.plot(ET, label='Evapotranspiration (mm)', color='red',alpha=0.7)
plt.plot(Q_obs, label='Observed Runoff (mm)', color='green',alpha=0.7)
plt.title('Area-Averaged Precipitation, ET, and Runoff for 2001')
plt.xlabel('Time (hours)')
plt.ylabel('Water Flux (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'area_averaged_precip_et_runoff_2001.png'), dpi=300)
plt.clf()

## --- Part 2: Model setup and calibration ---

# Rainfall-runoff model (finite difference approximation)
def simulate_runoff(k, P, ET, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q_obs[0]
    
    for t in range(2, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt/k)
        Q_sim[t] = max(0,Q_t) # Ensure non-negative runoff

    return (Q_sim)

# Define the objective (KGE) function
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    #print interative matrix, if needed
    #print(r, alpha, beta, kge)
    return (kge, r, alpha, beta)

# Create the list of k values and run the model to get simulated runoff and performance index
k_testlist = np.arange(0.15, 0.3, 0.15)
#k_testlist = 0.15
Q_sim_all = np.empty([len(P), len(k_testlist)])
PerfIndex_all = np.empty([len(k_testlist), 5]) #for k, kge, r, alpha, beta

n=0
for k in k_testlist:
    Qsim = simulate_runoff(k, P, ET)
    Q_sim_all[:,n] = Qsim
    
    PerfIndex = kge(Q_obs, Qsim)
    PerfIndex_all[n,0] = k
    PerfIndex_all[n,1:] = PerfIndex
    n += 1

# Plotting simulated runoff for different k values
# Line plot
for i in range(len(k_testlist)):
    print(f"k = {PerfIndex_all[i,0]:.2f}, KGE = {PerfIndex_all[i,1]:.3f}, r = {PerfIndex_all[i,2]:.3f}, alpha = {PerfIndex_all[i,3]:.3f}, beta = {PerfIndex_all[i,4]:.3f}")
    plt.plot(Q_sim_all[:, i], label=f'Simulated Runoff (k={PerfIndex_all[i,0]:.2f})',alpha=0.7)
plt.plot(Q_obs, label='Observed Runoff',alpha=0.7)
plt.title('Simulated Runoff for Different k Values')
plt.xlabel('Time (hours)')
plt.ylabel('Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_runoff_different_k.png'), dpi=300)
plt.clf()
# Scatter plot
for i in range(len(k_testlist)):
    plt.scatter(Q_obs, Q_sim_all[:, i], label=f'Simulated Runoff (k={PerfIndex_all[i,0]:.2f})', alpha=0.7)
plt.plot(Q_obs, Q_obs, color='black', linestyle='--', label='1:1 Line')
plt.title('Simulated vs Observed Runoff for Different k Values')
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_vs_observed_runoff_different_k.png'), dpi=300)
plt.clf()

## KGE Optimization ##
# Objective function for optimization
def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET)
    kge_model = kge(Q_obs, Q_sim)
    return (1.0 - kge_model[0])

# Optimize k using KGE
res = opt.minimize_scalar(objective, bounds=(0.1, 2), args=(P, ET, Q_obs), method='bounded')
print(f"Optimization result: \n{res}")

# Best k value
best_k = res.x
Q_sim = simulate_runoff(best_k, P, ET)
print(f"Optimized k: {best_k:.3f}")
optimized_kge = kge(Q_obs, Q_sim)
print(f"Optimized KGE: {optimized_kge[0]:.3f}, r = {optimized_kge[1]:.3f}, alpha = {optimized_kge[2]:.3f}, beta = {optimized_kge[3]:.3f}")
# Plotting the best k value
plt.plot(Q_sim, label=f'Simulated Runoff (k={best_k:.2f})',alpha=0.7)
plt.plot(Q_obs, label='Observed Runoff',alpha=0.7)
plt.title('Simulated and Observed Runoff for 2001 (Optimized k)')
plt.xlabel('Time (hours)')
plt.ylabel('Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_runoff_optimized.png'), dpi=300)
plt.clf()
# Scatter plot
plt.scatter(Q_obs, Q_sim, label=f'Simulated Runoff (k={best_k:.2f})',alpha=0.7)
plt.plot(Q_obs, Q_obs, color='black', linestyle='--', label='1:1 Line')
plt.title('Simulated vs Observed Runoff for 2001 (Optimized k)')
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_vs_observed_runoff_optimized.png'), dpi=300)
plt.clf()

# --- Validation ---

# Load the NetCDF files for validation (precipitation, ET, runoff)
precip_fileVal = os.path.join(data_dir,"era5_OLR_2002_total_precipitation.nc")
et_fileVal = os.path.join(data_dir,"era5_OLR_2002_total_evaporation.nc")
runoff_fileVal = os.path.join(data_dir,"ambientera5_OLR_2002_total_runoff.nc")

P_gridVal = load_and_clip(precip_fileVal, "tp", gdf) * 1000.0
ET_gridVal = load_and_clip(et_fileVal, "e", gdf) * 1000.0
Q_gridVal = load_and_clip(runoff_fileVal, "ro", gdf) * 1000.0

# Compute area-averaged values
P_v = P_gridVal.mean(dim=["latitude", "longitude"]).values
ET_v = ET_gridVal.mean(dim=["latitude", "longitude"]).values
Q_obs_v = Q_gridVal.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET_v = np.where(ET_v < 0.0, -ET_v, ET_v) 

# Plotting
plt.plot(P_v, label='Precipitation (mm)', color='blue',alpha=0.7)
plt.plot(ET_v, label='Evapotranspiration (mm)', color='red',alpha=0.7)
plt.plot(Q_obs_v, label='Observed Runoff (mm)', color='green',alpha=0.7)
plt.title('Area-Averaged Precipitation, ET, and Runoff for 2002')
plt.xlabel('Time (hours)')
plt.ylabel('Water Flux (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'area_averaged_precip_et_runoff_2002.png'), dpi=300)
plt.clf()
Q_sim_v = simulate_runoff(best_k, P_v, ET_v)
kge_v = kge(Q_obs_v, Q_sim_v)
# Plotting simulated runoff for different k values
# Line plot
plt.plot(Q_sim_v, label=f'Simulated Runoff (k={best_k:.2f})',alpha=0.7)
plt.plot(Q_obs_v, label='Observed Runoff',alpha=0.7)
plt.title('Simulated and Observed Runoff (Validation) for 2002')
plt.xlabel('Time (hours)')
plt.ylabel('Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_runoff_validation.png'), dpi=300)
plt.clf()
# Scatter plot
plt.scatter(Q_obs_v, Q_sim_v, label=f'Simulated Runoff (k={best_k:.2f})',alpha=0.7)
plt.plot(Q_obs_v, Q_obs_v, color='black', linestyle='--', label='1:1 Line')
plt.title('Simulated vs Observed Runoff (Validation) for 2002')
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff (mm)')
plt.legend()
plt.grid()
plt.savefig(os.path.join(fig_dir, 'simulated_vs_observed_runoff_validation.png'), dpi=300)
plt.clf()
print(f"Validation KGE: {kge_v[0]:.3f}, r = {kge_v[1]:.3f}, alpha = {kge_v[2]:.3f}, beta = {kge_v[3]:.3f}")
