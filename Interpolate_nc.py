import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load the data
f_base = 'N:/Planet-A/LUF-Modelling/LUTO2_JZ/biodiversity_data/data'
all_tifs = pd.read_csv('data/all_suitability_tifs.csv')



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
select = {'group': ['amphibians'], 
          'species': ['Arenophryne_rotunda'], 
          'ssp': ['ssp245'], 
          }



f_nc = xr.open_dataset(f"{f_base}/bio_BCC.CSM2.MR.nc", engine='h5netcdf', chunks='auto')['data'].astype('int8')

f_nc_sel = f_nc.sel(**select)
f_nc_year = f_nc_sel.interp(year=[2030], 
                            method='linear', 
                            kwargs={"fill_value": "extrapolate"},
                            x=np.linspace(f_nc.x.min(), f_nc.x.max(), len(f_nc.x) * 5),
                            y=np.linspace(f_nc.y.min(), f_nc.y.max(), len(f_nc.y) * 5)
            )

f_nc_coords = f_nc_year.sel(x=coord_sparse[0], y=coord_sparse[1], method='nearest')


plt.scatter(f_nc_coords.x, f_nc_coords.y, c=f_nc_coords.values.flatten(), cmap='viridis', s=0.01)