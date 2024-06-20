import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed
from codes import bincount_avg, get_id_map_by_upsample_reproject

# Parameters
max_worker = 10

# Reference data
NLUM = rxr.open_rasterio('data/NLUM_2010-11_mask.tif', chunks='auto').squeeze('band').drop_vars('band').astype(np.bool_)
bio_map = rxr.open_rasterio('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif', chunks='auto').squeeze('band').drop_vars('band')
bio_map = bio_map.rio.write_crs(NLUM.rio.crs)

# Load the cell_df that contains the land use category and description for each cell
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
lumap = lumap.rename(columns={'X': 'x', 'Y': 'y'})

lumap_xr = xr.Dataset.from_dataframe(lumap.set_index(['PRIMARY_V7','LU_DESC','y','x'])[['LU_ID']]).chunk('auto')['LU_ID'] 
lumap_xr = lumap_xr.sortby('PRIMARY_V7').sortby('LU_DESC')                  
lumap_xr = lumap_xr.reindex_like(NLUM, tolerance=0.001, method='nearest')       # fill the missing values with the nearest ones
lumap_xr = xr.where(lumap_xr.isnull(), 0, 1).astype(np.bool_)                   # Convert the LU_ID to binary mask
lumap_xr = lumap_xr.rio.write_crs(NLUM.rio.crs)
lumap_xr = lumap_xr.rio.write_transform(NLUM.rio.transform())


# Sanity check: Make sure the non-ag and ag cells are added up to the total cells
non_ag_xr = lumap_xr.sel(LU_DESC=['Non-agricultural land'])
ag_xr = lumap_xr.sel(LU_DESC=[i for i in lumap_xr.LU_DESC.values if i != 'Non-agricultural land'])

num_non_ag = non_ag_xr.sum().values
num_ag = ag_xr.sum().values
num_total = NLUM.sum().values
if num_non_ag + num_ag != num_total:
    raise ValueError(f'The sum of `non-ag cells` ({num_non_ag}) and `ag cells` {num_ag} is not equal to the `total cells` ({num_total}).')


# Reproject and resample the non-ag map to the bio_map resolution 
mask_id = get_id_map_by_upsample_reproject(bio_map, NLUM, NLUM.rio.crs, bio_map.rio.transform()) # Upsample and reproject bio_map to the same CRS and resolution as NLUM

tasks = []
for i,j in product(sorted(lumap_xr['PRIMARY_V7'].values), sorted(lumap_xr['LU_DESC'].values)):
    arr_selected = lumap_xr.sel(PRIMARY_V7=i, LU_DESC=j)    # Select by list will loss indexing info for combining
    arr_selected = arr_selected.expand_dims({'PRIMARY_V7': [i], 'LU_DESC': [j]})
    tasks.append(delayed(bincount_avg)(mask_id, arr_selected, bio_map))

para_obj = Parallel(n_jobs=min(len(tasks), max_worker), return_as='generator')
lumap_xr = xr.combine_by_coords([i for i in para_obj(tqdm(tasks, total=len(tasks)))])



# Save to nc   
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}} 
lumap_xr.name = 'data'
lumap_xr.to_netcdf(f'data/lumap_2d_all_lucc_5km.nc', mode='w', encoding=encoding, engine='h5netcdf')




# Sanity check
if __name__ == '__main__':
    
    # Save lumap_1km to tif
    ag_xr.sum('PRIMARY_V7').astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/ag_mask.tif')
    ag_xr.sum(['PRIMARY_V7','LU_DESC']).astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/ag_mask_sum.tif')
    
    non_ag_xr.sum('PRIMARY_V7').astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask.tif')
    non_ag_xr.sum(['PRIMARY_V7','LU_DESC']).astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_sum.tif')

    # Save the reprojected lumap_5km to tif
    lumap_xr = xr.open_dataset('data/lumap_2d_all_lucc_5km.nc')['data']
    
    ag_xr_5km = lumap_xr.sel(LU_DESC=[i for i in lumap_xr.LU_DESC.values if i != 'Non-agricultural land'])
    ag_xr_5km.sum('PRIMARY_V7').rio.write_nodata(-1).rio.to_raster('data/ag_mask_5km.tif')
    ag_xr_5km.sum(['PRIMARY_V7','LU_DESC']).rio.write_nodata(-1).rio.to_raster('data/ag_mask_sum_5km.tif')
    
    non_ag_xr_5km = lumap_xr.sel(LU_DESC=['Non-agricultural land'])
    non_ag_xr_5km.sum(['PRIMARY_V7']).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_5km.tif')
    non_ag_xr_5km.sum(['PRIMARY_V7','LU_DESC']).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_sum_5km.tif')


















