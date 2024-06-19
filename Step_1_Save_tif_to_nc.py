import os
import rioxarray as rxr
import xarray as xr
import pandas as pd

from itertools import product

from codes.helper_func import get_all_path, replace_mask_with_nearest
from tqdm.auto import tqdm
from glob import glob
from joblib import Parallel, delayed


# # Search for all tif files and save to csv
# bio_path_raw = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km'
# bio_path_condition = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-condition_20-year_snapshots_5km'
# get_all_path(bio_path_condition, 'data/bio_file_paths_condition.csv')



# Load the csv file
bio_nc_df = 'data/bio_file_paths_condition.csv' # ['data/bio_file_paths_raw.csv', 'data/bio_file_paths_condition.csv']
bio_nc_dir = 'data/bio_nc_raw' if 'raw' in bio_nc_df else 'data/bio_nc_condition'
dtype = 'uint8' if 'raw' in bio_nc_df else 'float32'
df = pd.read_csv(bio_nc_df)


# Get the mask for biodiversity maps
NLUM = rxr.open_rasterio('data/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8')
bio_sample = rxr.open_rasterio('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif').squeeze('band').drop_vars('band')
bio_sample = bio_sample.rio.write_crs(NLUM.rio.crs)
mask = bio_sample.values != bio_sample.rio.nodata if 'raw' in bio_nc_df else NLUM.rio.reproject_match(bio_sample, nodata=0).astype('bool')

# Filter out the ensemble data
ensemble_df = df.query('model == "GCM-Ensembles"').drop(columns=['group','model'])
valid_species = ensemble_df['species'].unique()                  

# Get the historical data. NOTE: Some species only have historical data, so we need to filter them out
historic_df = df.query('model == "historic" and species.isin(@valid_species) and ~path.str.contains("5x5")')




def process_row(row):
    ds = rxr.open_rasterio(row['path']).sel(band=1).drop_vars('band')
    ds.values = replace_mask_with_nearest(ds.values, ~mask).astype(dtype)
    return ds.expand_dims({'year':[row['year']], 'species':[row['species']]})


def save_to_nc(df, ssp, mode):
    # Check if the file already exists
    if os.path.exists(f'{bio_nc_dir}/{ssp}_{mode}.nc'):
        print(f'{ssp}_{mode}.nc already exists')
        return

    # Multi-threading to read TIF and expand dims
    in_df = df.query(f'ssp == "{ssp}" and mode == "{mode}"')
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
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": dtype}} 
    xr_chunk.to_netcdf(f'{bio_nc_dir}/{ssp}_{mode}.nc', mode='w', encoding=encoding, engine='h5netcdf')
                

        
# Save ensemble data to nc            
for ssp, mode in product(ensemble_df['ssp'].unique(), ensemble_df['mode'].unique()):
    save_to_nc(ensemble_df, ssp, mode)
    
# Save historic data to nc
save_to_nc(historic_df, 'historic', 'historic')
