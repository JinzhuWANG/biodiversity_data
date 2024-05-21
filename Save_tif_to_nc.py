import os
import numpy as np
import rioxarray as rxr
import xarray as xr
import pandas as pd

from glob import glob
from codes.helper_func import replace_with_nearest
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')
df = df.query('ssp != "historic"')
df['path'] = df['path'].str.replace('N:\\Data-Master\\', 'N:\\Planet-A\\Data-Master\\')


out_base = 'N:/Planet-A/LUF-Modelling/LUTO2_JZ/biodiversity_data/data'
exclud_model = ['GCM-Ensembles']

df = df.query('model not in @exclud_model')




def process_row(row, dims=None):
    ds = rxr.open_rasterio(row['path'])
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata)
    dims = {k: [v] for k, v in dict(row).items() if k != 'path'} if dims is None else dims
    return ds.expand_dims(dims)



# Loop through each model
for idx, group in df.groupby('model'):
    
    out_path = f'{out_base}/bio_{idx}.nc'
    encoding = {'data': {"compression": "gzip", "compression_opts": 9}}
    para_obj = Parallel(n_jobs=-1, prefer="threads", return_as='generator')
    tasks = (delayed(process_row)(row) for _, row in group.iterrows())
    pbar = tqdm(total=len(group))
    
    xr_chunk = []
    for result  in para_obj(tasks):
        xr_chunk.append(result)
        pbar.update(1)

    xr_chunk = xr.combine_by_coords(xr_chunk, fill_value=0)
    xr_chunk.name = 'data'
    xr_chunk.to_netcdf(out_path, mode='a', encoding=encoding, engine='h5netcdf')
        
    pbar.close()
