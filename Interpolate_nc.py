import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
ds = xr.open_dataset('data/ssp245_EnviroSuit.nc', engine='h5netcdf', chunks='auto')['data']


# Load index
index_dense = np.load("data/mask_idx_2d_dense.npy")
index_sparse = np.load("data/mask_idx_2d_sparse.npy")

NLUMASK = rxr.open_rasterio(f"data/NLUM_2010-11_mask.tif")
trans = NLUMASK.rio.transform()

def get_coord(index_ij):
    coord_x = trans.c + trans.a * index_ij[1]
    coord_y = trans.f + trans.e * index_ij[0]
    return xr.DataArray(coord_x, dims='points'), xr.DataArray(coord_y, dims='points')

coord_dense = get_coord(index_dense)
coord_sparse = get_coord(index_sparse)




# Get the nc file with adjusted chunk sizes
select = {
    'species': ['Arenophryne_rotunda'],
}


f_nc_sel = ds.sel(**select)
f_nc_year = f_nc_sel.interp(year=[2010], 
                            method='linear', 
                            kwargs={"fill_value": "extrapolate"},
                            x=np.linspace(ds.x.min(), ds.x.max(), len(ds.x) * 5),
                            y=np.linspace(ds.y.min(), ds.y.max(), len(ds.y) * 5)
            )

f_nc_coords = f_nc_year.sel(x=coord_sparse[0], y=coord_sparse[1], method='nearest')


plt.scatter(f_nc_coords.x, f_nc_coords.y, c=f_nc_coords.values.flatten(), cmap='viridis', s=0.01)