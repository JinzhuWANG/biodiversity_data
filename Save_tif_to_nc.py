
import os
import rioxarray as rxr
import xarray as xr
import pandas as pd

from itertools import product

from codes.helper_func import get_all_path, replace_with_nearest
from tqdm.auto import tqdm
from glob import glob
from joblib import Parallel, delayed


# # Search for all tif files and save to csv
# bio_path = r'N:\Data-Master\Biodiversity\Environmental-suitability\Annual-species-suitability_20-year_snapshots_5km'
# get_all_path(bio_path)

# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')
df = df.query('ssp != "historic"')

ensemble_df = df.query('model == "GCM-Ensembles"').drop(columns=['group','model'])


def process_row(row):
    ds = rxr.open_rasterio(row['path']).sel(band=1).drop_vars('band')
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata).astype('int8')
    return ds.expand_dims({'year':[row['year']], 'species':[row['species']]})



for ssp, mode in product(ensemble_df['ssp'].unique(), ensemble_df['mode'].unique()):
    # Filter the df
    in_df = ensemble_df.query(f'ssp == "{ssp}" and mode == "{mode}"')
 
    # Multi-threading to read TIF and expand dims
    para_obj = Parallel(n_jobs=-1, return_as='generator')
    tasks = (delayed(process_row)(row) for _, row in in_df.iterrows())
    pbar = tqdm(total=len(in_df))

    # Get results from the parallel processing
    xr_chunk = []
    for result  in para_obj(tasks):
        xr_chunk.append(result)
        pbar.update(1)
    pbar.close()

    # Combine the data and write to nc
    xr_chunk = xr.combine_by_coords(xr_chunk, fill_value=0, combine_attrs='drop')
    xr_chunk.name = 'data'
 
    # Save to nc   
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'int8'}} 
    xr_chunk.to_netcdf(f'data/{ssp}_{mode}.nc', mode='w', encoding=encoding, engine='h5netcdf')
            


            
            
