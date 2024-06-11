import xarray as xr
import numpy as np

from tqdm.auto import tqdm
from joblib import Parallel, delayed


# Load the data
ds_enssemble = xr.open_dataset('data/ssp245_EnviroSuit.nc', engine='h5netcdf', chunks='auto')['data']
ds_historic = xr.open_dataset('data/historic_historic.nc', engine='h5netcdf', chunks='auto')['data']

# Split the ds into chunks at the species dimension
chunk_size = 30
species_num = len(ds_enssemble['species'])
species_chunks = np.array_split(np.arange(species_num), chunk_size)

# Fix the rounding error for x/y coords between the two datasets
ds_historic['y'] = ds_enssemble['y']
ds_historic['x'] = ds_enssemble['x']
ds = xr.combine_by_coords([ds_historic, ds_enssemble])['data']


# Load index
coord_lon_lat = np.load("data/coord_lon_lat_res1.npy")
coord_x = xr.DataArray(coord_lon_lat[0], dims='points')
coord_y = xr.DataArray(coord_lon_lat[1], dims='points')


def interp_sel(ds, up_scale=5, coord_x=coord_x, coord_y=coord_y):
    # Interpolate, filter and round
    ds_interp = ds.interp(
        year=[2030], 
        x=np.linspace(ds.x.min(), ds.x.max(), len(ds.x) * up_scale),
        y=np.linspace(ds.y.min(), ds.y.max(), len(ds.y) * up_scale),
        method='linear', 
        kwargs={"fill_value": "extrapolate"},
    ).round().astype(np.int8)
    
    # Select the points
    ds_sel = ds_interp.sel(
        x=coord_x, 
        y=coord_y, 
        method='nearest')

    return ds_sel.compute()


# Create tasks 
tasks = [delayed(interp_sel)(ds.isel(species=species_chunk)) for species_chunk in species_chunks]
para_obj = Parallel(n_jobs=chunk_size, return_as='generator')

# Parallel processing to get the interpolated values
val_pts = []
for out in tqdm(para_obj(tasks), total=len(species_chunks)):
    val_pts.append(out)

val_pts = xr.combine_by_coords(val_pts)
