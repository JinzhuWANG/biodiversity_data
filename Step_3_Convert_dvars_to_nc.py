
import numpy as np
import rioxarray as rxr
import xarray as xr

from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed

from codes import ag_dvar_to_bio_map, combine_future_hist, non_ag_dvar_to_bio_map, am_dvar_to_bio_map, non_ag_to_xr, ag_to_xr, am_to_xr
from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

# Read in fake data
max_workers = 10
res_factor = 5
year = 2050
data = Data(res_factor)


# Get dvars
ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{year}.npy')         
am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{year}.npy')  
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}
non_ag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{year}.npy')


# dvar to xarray 
ag_dvar = ag_to_xr(data, ag_dvar)
am_dvar = am_to_xr(data, am_dvar)
non_ag_dvar = non_ag_to_xr(data, non_ag_dvar)



# Reproject and match dvars to the bio map
ag_dvar_map_reprojected = ag_dvar_to_bio_map(data, ag_dvar, res_factor, max_workers)
am_dvar_map_reprojected = am_dvar_to_bio_map(data, am_dvar, res_factor, max_workers)
non_ag_dvar_map_reprojected = non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, max_workers)



# Get the biodiversity data
bio_xr_mask = xr.open_dataarray('data/bio_mask.nc', chunks='auto').astype(np.bool_)
bio_xr_raw = combine_future_hist('data/bio_nc_raw/ssp245_EnviroSuit.nc', 'data/bio_nc_raw/historic_historic.nc')
bio_xr_raw = bio_xr_raw.reindex_like(bio_xr_mask, method='nearest')                 # Make the data have the same x/y coordinates to mask, so thery are spatially aligned
bio_xr_hist_sum = (bio_xr_raw.sel(year=1990) * bio_xr_mask).sum(['x', 'y'])         # Get the sum of all historic cells 
bio_xr_contribution = (bio_xr_raw / bio_xr_hist_sum ).astype(np.float32)*100        # Calculate the contribution of each cell (%) to the total biodiversity


# Calculate the biodiversity contribution of each cell
workers = 50
para_obj = Parallel(n_jobs=workers, return_as='generator')
interp_year = [2040]

def interp_by_year(ds, year):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()

chunks_idx = np.array_split(range(bio_xr_contribution['species'].size), workers)
bio_xr_chunks = [bio_xr_contribution.isel(species=idx) for idx in chunks_idx]
bio_xr_tasks = [delayed(interp_by_year)(chunk, interp_year) for chunk in bio_xr_chunks]
bio_xr_interp = xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_tasks), total=len(bio_xr_tasks))])['data']






# Sanity check
if __name__ == '__main__':
    
    NLUM = rxr.open_rasterio('data/NLUM_2010-11_mask.tif')
    
    maps = []
    for lm,lu in product(ag_dvar['lm'].values, ag_dvar['lu'].values):
        map_ = ag_dvar.sel(lm=lm, lu=lu).values
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_)
        map_ = xr.DataArray(
            empty_map,
            dims=('y', 'x'),
            coords={'y': NLUM['y'].values, 'x': NLUM['x'].values})
        map_ = map_.expand_dims({'lm': [lm], 'lu': [lu]})
        maps.append(map_)
        
    map_ = xr.combine_by_coords(maps)
    map_ = map_.where(map_>=0, 0)
    map_.rio.write_crs(NLUM.rio.crs, inplace=True)
    map_.rio.write_transform(NLUM.rio.transform(), inplace=True)
    map_.sum('lm').rio.to_raster('data/ag_dvar_map_1km.tif', compress='lzw')
    
    
    ag_dvar_map_reprojected.sum('lm').rio.to_raster('data/ag_dvar_map_5km.tif', compress='lzw')
    

