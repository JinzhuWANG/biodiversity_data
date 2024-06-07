import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
ds = xr.open_dataset('data/ssp245_EnviroSuit.nc', engine='h5netcdf', chunks='auto')['data']


# Load index
index_dense = np.load("data/coord_lon_lat.npy")

coord_x = xr.DataArray(index_dense[0], dims='points')
coord_y = xr.DataArray(index_dense[1], dims='points')



# Get the nc file with adjusted chunk sizes
ds_species = ds.sel(species=['Abelmoschus_ficulneus'])

ds_interp = ds_species.interp(
    year=[2010], 
    x=np.linspace(ds.x.min(), ds.x.max(), len(ds.x) * 5),
    y=np.linspace(ds.y.min(), ds.y.max(), len(ds.y) * 5),
    method='linear', 
    kwargs={"fill_value": "extrapolate"},
)

ds_sel = ds_interp.sel(
    x=coord_x, 
    y=coord_y, 
    method='nearest')


plt.scatter(coord_x, coord_y, c=ds_sel.values.flatten(), cmap='viridis', s=0.01)