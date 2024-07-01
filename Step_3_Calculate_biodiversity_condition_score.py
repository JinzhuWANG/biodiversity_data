import os
import numpy as np
import xarray as xr
import luto.settings as settings

from glob import glob
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from codes import calc_bio_hist_sum, calc_bio_score_group, calc_bio_score_species


# Read in fake data
max_workers = 50
interp_year = [2040]
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}}
para_obj = Parallel(n_jobs=max_workers, return_as='generator')




  
# Helper functions to interpolate the biodiversity scores
def interp_by_year(ds, year):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()

def interp_bio_species(bio_contribution_species, interp_year):
    chunks_species = np.array_split(range(bio_contribution_species['species'].size), max_workers)
    bio_xr_chunks = [bio_contribution_species.isel(species=idx) for idx in chunks_species]
    bio_xr_tasks = [delayed(interp_by_year)(chunk, [year]) for chunk in bio_xr_chunks for year in interp_year]
    return xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_tasks), total=len(bio_xr_tasks))])['data']

def interp_bio_group(bio_contribution_group, interp_year):
    bio_xr_tasks_group = [delayed(interp_by_year)(bio_contribution_group, [year]) for year in interp_year]
    return xr.combine_by_coords([out for out in tqdm(para_obj(bio_xr_tasks_group), total=len(bio_xr_tasks_group))])['data']




# Save nc to disk
bio_raw_ncs = glob(f'{settings.INPUT_DIR}/*EnviroSuit.nc')
for nc in bio_raw_ncs: 
    bio_his_score_sum = calc_bio_hist_sum(nc)
    fname = os.path.basename(nc).replace('EnviroSuit', 'Condition').replace('.nc', '')

    # Calculate the biodiversity contribution scores for each group between 2010 and 2100
    bio_xr_contribution_group = calc_bio_score_group(nc, bio_his_score_sum)
    
    if os.path.exists(f'{settings.INPUT_DIR}/{fname}_group.nc'):
        print(f'{fname}_group.nc already exists.')
        continue
    else:
        bio_xr_interp_group = interp_bio_group(bio_xr_contribution_group, range(2010,2101))
        bio_xr_interp_group.to_netcdf(f'{settings.INPUT_DIR}/{fname}_group.nc', mode='w', encoding=encoding, engine='h5netcdf')
    
    # 2.5T mem, 99% CPU, 10 hrs, ~450GB disk
    # bio_xr_contribution_species = calc_bio_score_species(nc, bio_his_score_sum)
    # bio_xr_interp_species = interp_bio_species(bio_xr_contribution_species, interp_year)
    # bio_xr_interp_species.to_netcdf(f'{settings.INPUT_DIR}/{fname}_species.nc', mode='w', encoding=encoding, engine='h5netcdf')


    

