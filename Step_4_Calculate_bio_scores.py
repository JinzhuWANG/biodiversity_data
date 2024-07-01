import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

import luto.settings as settings

from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed
from codes.fake_func import Data
from codes import ag_dvar_to_bio_map, calc_bio_hist_sum, calc_bio_score_species, non_ag_dvar_to_bio_map, am_dvar_to_bio_map, non_ag_to_xr, ag_to_xr, am_to_xr
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES


res_factor = 5
workers = 15
dvar_year = 2050
interp_year = [dvar_year]
data = Data(res_factor)
para_obj = Parallel(n_jobs=workers, return_as='generator')



# Get dvars; Pretend loading dvars in the LUTO project
ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{dvar_year}.npy')         
non_ag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{dvar_year}.npy')
am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{dvar_year}.npy')  
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}

ag_dvar = ag_to_xr(data, ag_dvar)
am_dvar = am_to_xr(data, am_dvar)
non_ag_dvar = non_ag_to_xr(data, non_ag_dvar)

# Reproject and match dvars to the bio map
ag_dvar = ag_dvar_to_bio_map(data, ag_dvar, res_factor, para_obj).chunk('auto')
am_dvar = am_dvar_to_bio_map(data, am_dvar, res_factor, para_obj).chunk('auto')
non_ag_dvar = non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, para_obj).chunk('auto')



# Calculate the biodiversity contribution scores for each group
bio_contribution_group = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_Condition_group.nc', chunks='auto') 

def xr_to_df(xr):
    return xr.sum(['x', 'y']).to_dataframe(name='contribution_%').reset_index()

def cal_bio_score_by_yr(dvar, year, para_obj=para_obj):
    bio_score_by_yr = [bio_contribution_group.sel(year=year, group=group)*dvar for group in bio_contribution_group['group'].values]
    tasks = [delayed(xr_to_df)(bio_score) for bio_score in bio_score_by_yr]
    out = pd.concat([out for out in tqdm(para_obj(tasks), total=len(tasks))], ignore_index=True)
    return out.drop(columns=['spatial_ref'])


ag_bio_score = cal_bio_score_by_yr(ag_dvar, interp_year)
am_bio_score = cal_bio_score_by_yr(am_dvar, interp_year)
non_ag_bio_score = cal_bio_score_by_yr(non_ag_dvar, interp_year)


# Calculate the biodiversity contribution scores for each species
species_split_size = 50

def interp_by_year(ds, year):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()

def interp_bio_species(bio_contribution_species, interp_year):
    chunks_species = np.array_split(range(bio_contribution_species['species'].size), species_split_size)
    bio_xr_chunks = [bio_contribution_species.isel(species=idx) for idx in chunks_species]
    bio_xr_tasks = [delayed(interp_by_year)(chunk, [year]) for chunk in bio_xr_chunks for year in interp_year]
    return xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_tasks), total=len(bio_xr_tasks))])['data']


bio_raw_path = f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_EnviroSuit.nc'
bio_his_score_sum = bio_his_score_sum = calc_bio_hist_sum(bio_raw_path)
bio_xr_contribution_species = calc_bio_score_species(bio_raw_path, bio_his_score_sum)
bio_xr_interp_species = interp_bio_species(bio_xr_contribution_species, interp_year)




# Sanity check
if __name__ == '__main__':
    
    NLUM = rxr.open_rasterio('input/NLUM_2010-11_mask.tif')
    
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
    
    
    ag_dvar.sum('lm').rio.to_raster('data/ag_dvar_map_5km.tif', compress='lzw')


