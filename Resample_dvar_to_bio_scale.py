import rioxarray as rxr
import xarray as xr
import numpy as np
import pandas as pd

from itertools import product
from joblib import Parallel, delayed
from codes import ag_to_xr, am_to_xr, bincount_avg, get_id_map_by_upsample_reproject, match_lumap_biomap, non_ag_to_xr
from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


res_factor = 10
year = 2050
data = Data(res_factor)

# Reference data
NLUM = rxr.open_rasterio('data/NLUM_2010-11_mask.tif', chunks='auto').squeeze('band').drop_vars('band').astype(np.bool_)
bio_map = rxr.open_rasterio('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif', chunks='auto').squeeze('band').drop_vars('band')
bio_map = bio_map.rio.write_crs(NLUM.rio.crs)

# Load the cell_df that contains the land use category and description for each cell
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
lumap = lumap.rename(columns={'X': 'x', 'Y': 'y'}).reset_index(drop=True) 

lumap_xr = xr.Dataset.from_dataframe(lumap.set_index(['PRIMARY_V7','y','x'])[['LU_ID']]).chunk('auto')['LU_ID']
lumap_xr = lumap_xr.reindex_like(NLUM, tolerance=0.001, method='nearest')       # fill the missing values with the nearest ones
lumap_xr = lumap_xr.rio.write_crs(NLUM.rio.crs)
lumap_xr = lumap_xr.rio.write_transform(NLUM.rio.transform())

non_ag_map = xr.where(lumap_xr==0, 1, 0).astype(np.float32)

# Sanity check: Make sure the non-ag and ag cells are added up to the total cells
num_non_ag = non_ag_map.sum().values
num_ag = (lumap_xr>0).sum().values
num_total = NLUM.sum().values
if num_non_ag + num_ag != num_total:
    raise ValueError(f'The sum of `non-ag cells` ({num_non_ag}) and `ag cells` {num_ag} is not equal to the `total cells` ({num_total}).')


# Reproject and resample the non-ag map to the bio_map resolution 
mask_id = get_id_map_by_upsample_reproject(bio_map, NLUM, NLUM.rio.crs, bio_map.rio.transform()) # Upsample and reproject bio_map to the same CRS and resolution as NLUM
tasks = [delayed(bincount_avg)(mask_id, non_ag_map.sel(PRIMARY_V7=[i]), bio_map) for i in non_ag_map['PRIMARY_V7'].values] 
para_obj = Parallel(n_jobs=min(len(tasks), 50))
matched_map = xr.combine_by_coords(para_obj(tasks))











ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{year}.npy')          # mrj
ag_dvar = ag_to_xr(data, ag_dvar)
ag_dvar = ag_dvar.reindex(lu=data.AGRICULTURAL_LANDUSES)


am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{year}.npy') 
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}
am_dvar = xr.combine_by_coords([am_to_xr(data, k, v) for k,v in am_dvar.items()])
am_dvar = am_dvar.reindex(am=AG_MANAGEMENTS_TO_LAND_USES.keys())


nonag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{year}.npy')
nonag_dvar = non_ag_to_xr(data, nonag_dvar)
nonag_dvar = nonag_dvar.reindex(lu=data.NON_AGRICULTURAL_LANDUSES)





map_2d = []
for i,j in product(ag_dvar['lm'].values, ag_dvar['lu'].values):
    map_ = ag_dvar.sel(lm=i, lu=j)
    map_ = match_lumap_biomap(data, map_, res_factor)
    map_ = map_.expand_dims({'lm': [i], 'lu': [j]})
    map_2d.append(map_)
map_2d = xr.combine_by_coords(map_2d)



















