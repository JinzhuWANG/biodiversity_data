import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from dask.diagnostics import ProgressBar


# Load the data
ds_enssemble = xr.open_dataset('data/ssp245_EnviroSuit.nc', engine='h5netcdf', chunks='auto')['data']
ds_historic = xr.open_dataset('data/historic_historic.nc', engine='h5netcdf', chunks='auto')['data']

# Fix the rounding error for x/y coords between the two datasets
ds_historic['y'] = ds_enssemble['y']
ds_historic['x'] = ds_enssemble['x']
ds = xr.combine_by_coords([ds_historic, ds_enssemble])['data']

# Load index
coord_lon_lat = np.load("data/coord_lon_lat_res1.npy")
coord_x = xr.DataArray(coord_lon_lat[0], dims='points')
coord_y = xr.DataArray(coord_lon_lat[1], dims='points')


# Interpolate, filter and round
ds_interp = ds.interp(
    year=[2030], 
    x=np.linspace(ds.x.min(), ds.x.max(), len(ds.x) * 5),
    y=np.linspace(ds.y.min(), ds.y.max(), len(ds.y) * 5),
    method='linear', 
    kwargs={"fill_value": "extrapolate"},
).round().astype(np.int8)

ds_sel = ds_interp.sel(
    x=coord_x, 
    y=coord_y, 
    method='nearest')


# Get the values
with ProgressBar():
    val = ds_sel.data.compute()

# plt.scatter(coord_x, coord_y, c=ds_sel.values.flatten(), cmap='viridis', s=0.01)