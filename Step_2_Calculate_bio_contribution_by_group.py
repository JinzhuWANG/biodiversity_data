'''
Codes here are used to calculate the biodiversity contribution by group.
The process includes:

1. Search for biodiversity NetCDF files
2. Calculate the historical (1990) biodiversity score sum for each species
3. Calculate the biodiversity contribution scores for each group between 2010 and 2100
4. Save the biodiversity contribution scores to disk


Key concepts:
    - The biodiversity contribution score is calculated as `species_layer[2010, 2011, ..., 2100] / sum(species_layer_1990)`.


Author: Jinzhu WANG
Data:   5 July 2024
Email:  wangjinzhulala@gmail.com

'''


import os
import xarray as xr

from glob import glob
from tqdm.auto import tqdm
from joblib import Parallel
from codes import calc_bio_hist_sum, calc_bio_score_group, interp_bio_group_to_shards


# Search for biodiversity NetCDF files
max_workers = 30        # ~70% utilization for a 256-core CPU
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}}
para_obj = Parallel(n_jobs=max_workers, return_as='generator')
bio_raw_ncs = glob('data/*EnviroSuit.nc')

  
# Save nc to disk
for nc in bio_raw_ncs:
    fname = os.path.basename(nc).replace('EnviroSuit', 'Condition').replace('.nc', '')
    # Calculate the historical biodiversity score sum for each species
    bio_his_score_sum = calc_bio_hist_sum(nc)
    # Calculate the biodiversity contribution scores for each group between 2010 and 2100
    if os.path.exists(f'data/{fname}_group.nc'):
        print(f'{fname}_group.nc already exists.')
        continue
    else:
        bio_xr_contribution_group = calc_bio_score_group(nc, bio_his_score_sum)
        bio_xr_interp_group = interp_bio_group_to_shards(bio_xr_contribution_group, range(2010,2101))
        bio_xr_interp_group = xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_interp_group), total=len(bio_xr_interp_group))])['data']
        bio_xr_interp_group.to_netcdf(f'data/{fname}_group.nc', mode='w', encoding=encoding, engine='h5netcdf')

