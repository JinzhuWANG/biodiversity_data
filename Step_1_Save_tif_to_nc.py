import os
import rioxarray as rxr
import xarray as xr
import pandas as pd
import luto.settings as settings

from itertools import product
from codes.helper_func import get_all_path, replace_mask_with_nearest
from tqdm.auto import tqdm
from glob import glob
from joblib import Parallel, delayed


# Search for all tif files and save to their paths to a csv file
bio_path_raw = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km'
if not os.path.exists('data/bio_file_paths_raw.csv'):
    get_all_path(bio_path_raw, 'data/bio_file_paths_condition.csv')
df = pd.read_csv('data/bio_file_paths_raw.csv' )


# Create an tempalate nc file for biodiversity data
NLUM = rxr.open_rasterio(f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8')
bio_mask = rxr.open_rasterio(df.iloc[0]['path']).squeeze('band').drop_vars('band').astype('uint8')
bio_mask = xr.where(bio_mask != bio_mask.rio.nodata, 1, 0).astype('uint8').chunk('auto')
bio_mask = bio_mask.rio.write_crs(NLUM.rio.crs)


bio_mask.name = 'data'
bio_mask.to_netcdf(
    f'{settings.INPUT_DIR}/bio_mask.nc', 
    mode='w', 
    encoding={'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}},
    engine='h5netcdf')




# Filter out the ensemble data
ensemble_df = df.query('model == "GCM-Ensembles" & mode == "EnviroSuit"').drop(columns=['model'])
valid_species = ensemble_df['species'].unique()                  
historic_df = df.query('model == "historic" and species.isin(@valid_species) and ~path.str.contains("5x5")')


# Define the function to convert tif to nc
def process_row(row):
    ds = rxr.open_rasterio(row['path']).sel(band=1).drop_vars('band')
    ds.values = replace_mask_with_nearest(ds.values, ~bio_mask.values).astype('uint8')
    ds = ds.expand_dims({'year':[row['year']], 'species':[row['species']], 'group':[row['group']]})
    ds['x'] = bio_mask['x']
    ds['y'] = bio_mask['y']
    return ds.squeeze('group')  # Drop the group dimension, but keep the group as a coordinate


def tif_to_nc(df, ssp, mode):
    # Multi-threading to read TIF and expand dims
    in_df = df.query(f'ssp == "{ssp}" and mode == "{mode}"')
    tasks = (delayed(process_row)(row) for _,row in in_df.iterrows())
    para_obj = Parallel(n_jobs=-1, return_as='generator')
    return [result for result in tqdm(para_obj(tasks), total=len(in_df))]

        
# Save ensemble data to nc      
historic_xr = tif_to_nc(historic_df, 'historic', 'historic')
for ssp, mode in product(ensemble_df['ssp'].unique(), ensemble_df['mode'].unique()):
    # Pass if the file already exists
    if os.path.exists(f'{settings.INPUT_DIR}/bio_{ssp}_{mode}.nc'):
        print(f'{ssp}_{mode}.nc already exists')
        continue

    # get the data
    ensemble_arrs = tif_to_nc(ensemble_df, ssp, mode)
    ensemble_arrs = xr.combine_by_coords(historic_xr + ensemble_arrs, fill_value=0, combine_attrs='drop')

    # Save to nc
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}} 
    ensemble_arrs.name = 'data'
    ensemble_arrs.to_netcdf(f'{settings.INPUT_DIR}/bio_{ssp}_{mode}.nc', mode='w', encoding=encoding, engine='h5netcdf')
    
