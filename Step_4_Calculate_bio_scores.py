from itertools import product
import numpy as np
import sparse
from tqdm.auto import tqdm
import xarray as xr
import rioxarray as rxr

import luto.settings as settings

from joblib import Parallel, delayed
from codes.fake_func import Data, get_coarse2D_map, upsample_array
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from codes import (ag_to_xr, bincount_avg, calc_bio_score_by_yr, match_lumap_biomap, non_ag_to_xr, am_to_xr, 
                   ag_dvar_to_bio_map, non_ag_dvar_to_bio_map, am_dvar_to_bio_map,
                   calc_bio_hist_sum, calc_bio_score_species, interp_bio_species_to_shards)



# Make a fake LUTO data object
res_factor = 10
data = Data(res_factor)
interp_year = 2010

# Define parameters
workers = 3
dvar_year = 2050
para_obj = Parallel(n_jobs=workers, return_as='generator', backend='threading')


# Get dvars; Here pretend loading dvars from the LUTO solver
ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{dvar_year}.npy')         
non_ag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{dvar_year}.npy')
am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{dvar_year}.npy')  
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}

ag_dvar = ag_to_xr(data, ag_dvar)
am_dvar = am_to_xr(data, am_dvar)
non_ag_dvar = non_ag_to_xr(data, non_ag_dvar)


# Reproject and match dvars to the bio map. NOTE: The dvars are sparsed array at ~5km resolution.
ag_dvar = ag_dvar_to_bio_map(data, ag_dvar, res_factor, para_obj).chunk('auto').compute()
am_dvar = am_dvar_to_bio_map(data, am_dvar, res_factor, para_obj).chunk('auto').compute()
non_ag_dvar = non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, para_obj).chunk('auto').compute()




# Calculate the biodiversity contribution scores
if settings.BIO_CALC_LEVEL == 'group':
    bio_score_group = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_Condition_group.nc', chunks='auto')
    bio_score_all_species_mean = bio_score_group.mean('group').expand_dims({'group': ['all_species']})  # Calculate the mean score of all species
    bio_score_group = xr.combine_by_coords([bio_score_group, bio_score_all_species_mean])['data']       # Combine the mean score with the original score
    bio_contribution_shards = [bio_score_group.sel(year=interp_year, group=group) for group in bio_score_group['group'].values] 
elif settings.BIO_CALC_LEVEL == 'species':
    bio_raw_path = f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_EnviroSuit.nc'
    bio_his_score_sum = calc_bio_hist_sum(bio_raw_path)
    bio_contribution_species = calc_bio_score_species(bio_raw_path, bio_his_score_sum)
    bio_contribution_shards = interp_bio_species_to_shards(bio_contribution_species, interp_year)
else:
    raise ValueError('Invalid settings.BIO_CALC_LEVEL! Must be either "group" or "species".')


bio_df = calc_bio_score_by_yr(ag_dvar, am_dvar, non_ag_dvar, bio_contribution_shards, para_obj)








# Sanity check: Below are each step to calculate the biodiversity contribution scores
if __name__ == '__main__':
    
    # wrapper function for parallel processing
    def reproject_match_dvar(ag_dvar, lm, lu, res_factor):
        map_ = ag_dvar.sel(lm=lm, lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'lm': [lm], 'lu': [lu]})
        return map_


    tasks = [delayed(reproject_match_dvar)(ag_dvar, lm, lu, res_factor) 
                for lm,lu in product(ag_dvar['lm'].values, ag_dvar['lu'].values)]
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)


    bio_id_path:str=f'{settings.INPUT_DIR}/bio_id_map.nc'
    lumap_tempelate:str=f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif' 
    biomap_tempelate:str=f'{settings.INPUT_DIR}/bio_mask.nc'



    for lm,lu in tqdm(product(ag_dvar['lm'].values, ag_dvar['lu'].values)):
        map_ = ag_dvar.sel(lm=lm, lu=lu)

        NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
        bio_mask_ds = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', decode_coords="all")
        bio_map = bio_mask_ds['data']
        bio_map['spatial_ref'] = bio_mask_ds['spatial_ref']
        bio_id_map = xr.open_dataset(bio_id_path, chunks='auto')['data']
            
        if res_factor > 1:   
            map_ = get_coarse2D_map(data, map_)
            map_ = upsample_array(data, map_, res_factor)
        else:
            empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
            np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
            np.place(empty_map, empty_map >=0, map_.data.todense())
            map_ = empty_map
            
        map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': NLUM['y'], 'x': NLUM['x']})
        map_ = map_.where(map_>=0, 0)
        map_ = map_.rio.write_crs(NLUM.rio.crs)
        map_ = map_.rio.write_transform(NLUM.rio.transform())  
        
        map_ = bincount_avg(bio_id_map, map_,  bio_map)
        # map_ = map_.where(map_ != map_.rio.nodata, 0)
