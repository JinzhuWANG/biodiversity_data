import os
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import luto.settings as settings

from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed
from codes import bincount_avg, get_id_arr_gdf, get_id_map_by_upsample_reproject


# Parameters
max_worker = 10
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}} 

# Reference data
NLUM = rxr.open_rasterio(f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif', chunks='auto').squeeze('band').drop_vars('band').astype(np.bool_)
bio_mask_ds = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', decode_coords="all")
bio_mask = bio_mask_ds['data']
bio_mask['spatial_ref'] = bio_mask_ds['spatial_ref']

# Calculate the area of each cell in the bio_mask
if not os.path.exists(f'{settings.INPUT_DIR}/bio_cell_area_ha.nc'):
    cell_arr, cell_df = get_id_arr_gdf(bio_mask)
    cell_df_albers = cell_df.to_crs('EPSG:3577')                # GDA94 / Australian Albers
    cell_df_albers['area'] = cell_df_albers.area / 10000        # m2 to ha
    area_arr = cell_df_albers['area'].values.reshape(cell_arr.shape)
    area_arr = xr.DataArray(area_arr, dims=['y', 'x'], coords={'y': bio_mask['y'], 'x': bio_mask['x']})
    area_arr.name = 'data'
    area_arr.to_netcdf(
        f'{settings.INPUT_DIR}/bio_cell_area_ha.nc', 
        mode='w', 
        encoding={'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}}, engine='h5netcdf')


# Load the cell_df that contains the land use category and description for each cell
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
lumap = lumap.rename(columns={'X': 'x', 'Y': 'y'})

lumap_xr = xr.Dataset.from_dataframe(lumap.set_index(['PRIMARY_V7','LU_DESC','y','x'])[['LU_ID']]).chunk('auto')['LU_ID']                 
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


# Create an array of the same shape as the bio_mask, and each cell has the value of the chronological ID of the bio_mask
# then upsample and reproject the id array to the same resolution as the lumap, 
# and finally, each cell in the id_map will have the id of bio_mask but the CRS/resolution of lumap
id_map = get_id_map_by_upsample_reproject(bio_mask, NLUM) 



# Calculate the average of of lumap cells (~1km) within each mask_id cell (~5km)
tasks = []
for i,j in product(sorted(lumap_xr['PRIMARY_V7'].values), sorted(lumap_xr['LU_DESC'].values)):
    arr_selected = lumap_xr.sel(PRIMARY_V7=i, LU_DESC=j)
    arr_selected = arr_selected.expand_dims({'PRIMARY_V7': [i], 'LU_DESC': [j]})
    tasks.append(delayed(bincount_avg)(id_map, arr_selected, bio_mask))

para_obj = Parallel(n_jobs=min(len(tasks), max_worker), return_as='generator')
lumap_xr = xr.combine_by_coords([i for i in para_obj(tqdm(tasks, total=len(tasks)))])

ag_xr_5km = lumap_xr.sel(LU_DESC=[i for i in lumap_xr.LU_DESC.values if i != 'Non-agricultural land']).sum(['PRIMARY_V7'])
non_ag_xr_5km = lumap_xr.sel(LU_DESC=['Non-agricultural land']).sum(['LU_DESC'])


# Save to nc   
lumap_xr.name = 'data'
ag_xr_5km.name = 'data'
non_ag_xr_5km.name = 'data'

lumap_xr.to_netcdf('data/lumap_2d_all_lucc_5km.nc', mode='w', encoding=encoding, engine='h5netcdf')
ag_xr_5km.to_netcdf('data/lumap_2d_all_lucc_5km_ag.nc', mode='w', encoding=encoding, engine='h5netcdf')
non_ag_xr_5km.to_netcdf('data/lumap_2d_all_lucc_5km_non_ag.nc', mode='w', encoding=encoding, engine='h5netcdf')



# Sanity check
if __name__ == '__main__':
    
    # Save lumap_1km to tif
    ag_xr.sum('PRIMARY_V7').astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/ag_mask.tif')
    ag_xr.sum(['PRIMARY_V7','LU_DESC']).astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/ag_mask_sum.tif')
    non_ag_xr.sum('LU_DESC').astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask.tif')
    non_ag_xr.sum(['PRIMARY_V7','LU_DESC']).astype(np.int8).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_sum.tif')

    # Save the reprojected lumap_5km to tif
    lumap_xr = xr.open_dataset('data/lumap_2d_all_lucc_5km.nc', chunks='auto')['data']

    ag_xr_5km.rio.write_nodata(-1).rio.to_raster('data/ag_mask_5km.tif')
    ag_xr_5km.sum('LU_DESC').rio.write_nodata(-1).rio.to_raster('data/ag_mask_sum_5km.tif')
    non_ag_xr_5km.rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_5km.tif')
    non_ag_xr_5km.sum('PRIMARY_V7',).rio.write_nodata(-1).rio.to_raster('data/non_ag_mask_sum_5km.tif')


















