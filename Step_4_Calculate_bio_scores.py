import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

import luto.settings as settings

from tqdm.auto import tqdm
from itertools import product
from joblib import Parallel, delayed
from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES
from codes import (ag_to_xr, non_ag_to_xr, am_to_xr, 
                   ag_dvar_to_bio_map, non_ag_dvar_to_bio_map, am_dvar_to_bio_map,
                   calc_bio_hist_sum, calc_bio_score_species, interp_bio_species_to_shards)



res_factor = 5
workers = 15
dvar_year = 2050

interp_year = [dvar_year]
data = Data(res_factor)
para_obj = Parallel(n_jobs=workers, return_as='generator')


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



# Helper functions to calculate the biodiversity contribution scores
def xr_to_df(shard, dvar_ag, dvar_am, dvar_non_ag):
    
    # Get the values to avoid duplicate computation
    if isinstance(shard, tuple):            # This means shard is a tuple of delayed function and arguments
        f, args = shard[0], shard[1]
        shard_xr = f(*args)
    elif isinstance(shard, xr.DataArray):   # This means shard is an xr.DataArray
        shard_xr = shard.compute()
    else:
        raise ValueError('Invalid shard type! Should be either tuple (delayed(function), args) or xr.DataArray.')

    # Calculate the biodiversity contribution scores
    tmp_ag = (shard_xr * dvar_ag).compute().sum(['x', 'y'])
    tmp_am = (shard_xr * dvar_am).compute().sum(['x', 'y'])
    tmp_non_ag = (shard_xr * dvar_non_ag).compute().sum(['x', 'y'])
    
    # Convert to dense array
    tmp_ag.data = tmp_ag.data.todense()
    tmp_am.data = tmp_am.data.todense()
    tmp_non_ag.data = tmp_non_ag.data.todense()
    
    # Convert to dataframe
    tmp_ag_df = tmp_ag.to_dataframe(name='contribution_%').reset_index()
    tmp_am_df = tmp_am.to_dataframe(name='contribution_%').reset_index()
    tmp_non_ag_df = tmp_non_ag.to_dataframe(name='contribution_%').reset_index()
    
    # Add land use type
    tmp_ag_df['lu_type'] = 'ag'
    tmp_am_df['lu_type'] = 'am'
    tmp_non_ag_df['lu_type'] = 'non_ag'
    
    return pd.concat([tmp_ag_df, tmp_am_df, tmp_non_ag_df], ignore_index=True)


def cal_bio_score_by_yr(bio_shards, interp_year, para_obj=para_obj):
    tasks = [delayed(xr_to_df)(bio_score, ag_dvar, am_dvar, non_ag_dvar) for bio_score in bio_shards]
    out = pd.concat([out for out in tqdm(para_obj(tasks), total=len(tasks))], ignore_index=True)
    return out.drop(columns=['spatial_ref'])



# Calculate the biodiversity contribution scores
if settings.BIO_CALC_LEVEL == 'group':
    bio_score_group = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_Condition_group.nc', chunks='auto')
    bio_contribution_shards = [bio_score_group.sel(year=interp_year, group=group) for group in bio_score_group['group'].values] 
elif settings.BIO_CALC_LEVEL == 'species':
    bio_raw_path = f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_EnviroSuit.nc'
    bio_his_score_sum = calc_bio_hist_sum(bio_raw_path)
    bio_contribution_species = calc_bio_score_species(bio_raw_path, bio_his_score_sum)
    bio_contribution_shards = interp_bio_species_to_shards(bio_contribution_species, interp_year)
else:
    raise ValueError('Invalid settings.BIO_CALC_LEVEL! Must be either "group" or "species".')



bio_df = cal_bio_score_by_yr(bio_contribution_shards[:2], interp_year)


# Export bio scores to geotiff for validation
if __name__ == '__main__':
    
    
    toy_bio_raw = np.random.randint(0,100,size=(4,5,5))
    toy_bio_contribution = toy_bio_raw / toy_bio_raw.sum()
    
    toy_bio_group = toy_bio_contribution.mean(axis=0)
    toy_bio_group.sum()
