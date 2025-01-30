import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

dset = xr.open_dataset(r'/mnt/storage1/maj0d/projects/data/N21E039.SRTMGL1_NC.nc')
DEM=np.array(dset.variables['SRTMGL1_DEM'])
plt.imshow(DEM)
cbar = plt.colorbar()
cbar.set_label('Elevation (m asl)')
plt.show()
plt.savefig('assignment_1.png',dpi=300)
