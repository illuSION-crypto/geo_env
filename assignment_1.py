import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

figures_dir = "/mnt/storage1/maj0d/projects/geo_env/figures/assignment_1"
dset = xr.open_dataset(r'/mnt/storage1/maj0d/projects/data/N21E039.SRTMGL1_NC.nc')
print(dset)
x_label = np.linspace(39,40, 10)
x_label = ["%.2f" % i for i in x_label]
y_label = np.linspace(22,21, 10)
y_label = ["%.2f" % i for i in y_label]
DEM=np.array(dset.variables['SRTMGL1_DEM'])
plt.imshow(DEM)
plt.xticks(np.linspace(0, dset.sizes['lon'], 10), x_label)
plt.xlabel('Longitude (degrees)')
plt.yticks(np.linspace(0, dset.sizes['lat'], 10), y_label)
plt.ylabel('Latitude (degrees)')
cbar = plt.colorbar()
cbar.set_label('Elevation (m asl)')
plt.title('Digital Elevation Model for Jeddah')
plt.show()
fig_name = 'assignment_1.png'
fig_path = os.path.join(figures_dir, fig_name)
plt.savefig(fig_path,dpi=300)
