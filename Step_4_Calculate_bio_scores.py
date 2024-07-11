
import os
import numpy as np
import sparse
from tqdm.auto import tqdm
import xarray as xr
import rioxarray as rxr
from codes import calc_bio_hist_sum, calc_bio_score_by_yr, calc_bio_score_species, interp_bio_species_to_shards
import luto.settings as settings


from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES



# Make a fake LUTO data object
res_factor = 5
data = Data(res_factor)
yr_cal = 1025



data.add_ag_dvars_xr(2010, data.ag_dvars[yr_cal])
data.add_am_dvars_xr(2015, data.ag_man_dvars[yr_cal])
data.add_non_ag_dvars_xr(2015, data.non_ag_dvars[yr_cal])



# Get the decision variables for the year and convert them to xarray
ag_dvar_reprj_to_bio = data.ag_dvars_2D_reproj_match[yr_cal]
am_dvar_reprj_to_bio = data.ag_man_dvars_2D_reproj_match[yr_cal]
non_ag_dvar_reprj_to_bio = data.non_ag_dvars_2D_reproj_match[yr_cal]
            
# Calculate the biodiversity contribution scores
if settings.BIO_CALC_LEVEL == 'group':
    bio_score_group = xr.open_dataarray(f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_Condition_group.nc', chunks='auto')
    bio_score_all_species_mean = bio_score_group.mean('group').expand_dims({'group': ['all_species']})  # Calculate the mean score of all species
    bio_score_group = xr.combine_by_coords([bio_score_group, bio_score_all_species_mean])['data']       # Combine the mean score with the original score
    bio_contribution_shards = [bio_score_group.sel(year=yr_cal, group=group) for group in bio_score_group['group'].values] 
elif settings.BIO_CALC_LEVEL == 'species':
    bio_raw_path = f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_EnviroSuit.nc'
    bio_his_score_sum = calc_bio_hist_sum(bio_raw_path)
    bio_contribution_species = calc_bio_score_species(bio_raw_path, bio_his_score_sum)
    bio_contribution_shards = interp_bio_species_to_shards(bio_contribution_species, yr_cal)
else:
    raise ValueError('Invalid settings.BIO_CALC_LEVEL! Must be either "group" or "species".')

# Write the biodiversity contribution to csv
bio_df = calc_bio_score_by_yr(
    ag_dvar_reprj_to_bio, 
    am_dvar_reprj_to_bio, 
    non_ag_dvar_reprj_to_bio, 
    bio_contribution_shards)
