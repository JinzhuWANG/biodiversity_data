import xarray as xr
import rioxarray as rxr
import numpy as np




# Load index
f_base = 'N:/Planet-A/LUF-Modelling/LUTO2_JZ/biodiversity_data/data/'
index_dense = np.load(f"{f_base}/mask_idx_2d_dense.npy")
index_sparse = np.load(f"{f_base}/mask_idx_2d_sparse.npy")


NLUMASK = rxr.open_rasterio(f"{f_base}/NLUM_2010-11_mask.tif")
trans = NLUMASK.rio.transform()

def get_coord(index_ij):
    coord_x = trans.c + trans.a * index_ij[1] 
    coord_y = trans.f + trans.e * index_ij[0]
    return coord_x, coord_y

coord_dense = get_coord(index_dense)
coord_sparse = get_coord(index_sparse)



# Get the nc file
f_nc = xr.open_dataset(f"{f_base}/bio_BCC.CSM2.MR.nc")
f_nc_species = f_nc.sel(species=['Acanthagenys_rufogularis', 'Acanthiza_ewingii', 'Acanthiza_inornata'])
f_nc_coords = f_nc_species.interp(
    year=[2010],
    x=coord_sparse[0], 
    y=coord_sparse[1], 
    method='linear',
    kwargs={"fill_value": "interpolate"})










