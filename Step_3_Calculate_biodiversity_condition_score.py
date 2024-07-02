import os
import xarray as xr
import luto.settings as settings

from glob import glob
from tqdm.auto import tqdm
from joblib import Parallel

from codes import calc_bio_hist_sum, calc_bio_score_group, interp_bio_group_to_shards


# Read in fake data
max_workers = 30        # ~70% CPU utilization
interp_year = [2040]
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}}
para_obj = Parallel(n_jobs=max_workers, return_as='generator')

bio_raw_ncs = glob(f'{settings.INPUT_DIR}/*EnviroSuit.nc')

  
# Save nc to disk
for nc in bio_raw_ncs:
    fname = os.path.basename(nc).replace('EnviroSuit', 'Condition').replace('.nc', '')
    # Calculate the historical biodiversity score sum for each species
    bio_his_score_sum = calc_bio_hist_sum(nc)
    # Calculate the biodiversity contribution scores for each group between 2010 and 2100
    if os.path.exists(f'{settings.INPUT_DIR}/{fname}_group.nc'):
        print(f'{fname}_group.nc already exists.')
        continue
    else:
        bio_xr_contribution_group = calc_bio_score_group(nc, bio_his_score_sum)
        bio_xr_interp_group = interp_bio_group_to_shards(bio_xr_contribution_group, range(2010,2101))
        bio_xr_interp_group = xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_interp_group), total=len(bio_xr_interp_group))])['data']
        bio_xr_interp_group.to_netcdf(f'{settings.INPUT_DIR}/{fname}_group.nc', mode='w', encoding=encoding, engine='h5netcdf')

