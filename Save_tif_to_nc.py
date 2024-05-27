import os
import rioxarray as rxr
import xarray as xr
import pandas as pd

from codes.helper_func import replace_with_nearest
from tqdm.auto import tqdm
from glob import glob
from joblib import Parallel, delayed

# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')
df = df.query('ssp != "historic"')
df = df.drop(columns=['group'])

exist_fs = glob('data/bio_*.nc')
out_base = 'data'



def process_row(row):
    ds = rxr.open_rasterio(row['path'])
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata).astype('int8')
    dims = {k:[v] for k,v in dict(row).items() if k not in ['path', 'model', 'mode']}
    return ds.expand_dims(dims)


def save_model(df, out_path, encoding):
    # Multi-threading to read TIF and expand dims
    para_obj = Parallel(n_jobs=30, prefer="threads", return_as='generator')
    tasks = (delayed(process_row)(row) for _, row in df.iterrows())
    pbar = tqdm(total=len(df))
    
    xr_chunk = []
    for result  in para_obj(tasks):
        xr_chunk.append(result)
        pbar.update(1)

    # Combine the data and write to nc
    xr_chunk = xr.combine_by_coords(xr_chunk, fill_value=0, combine_attrs='drop')
    xr_chunk.name = 'data'
    xr_chunk.to_netcdf(out_path, mode='w', encoding=encoding, engine='h5netcdf')
        
    pbar.close()




# Loop through each model
for idx, group in df.groupby(['model','mode']):
    
    out_name = '_'.join(idx)
    
    # Get the output path and dims
    out_path = f'{out_base}/bio_{out_name}.nc'
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'int8'}}
    
    # Check if the file already exists
    if os.path.exists(out_path):
        print(f'{out_name} already exists')
        continue
    
    # Save the model
    attempts = 0
    max_attempts = 10
    while not os.path.exists(out_path) and attempts < max_attempts:
        try:
            save_model(group, out_path, encoding)
        except Exception as e:
            print(f'Error in {(out_name)}: {e}')
            attempts += 1
            
            
            
