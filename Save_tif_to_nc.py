import numpy as np
import rioxarray as rxr
import xarray as xr
import pandas as pd

from codes.helper_func import replace_with_nearest
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')
df = df.query('ssp != "historic"')
df['path'] = df['path'].str.replace('N:\\Data-Master\\', 'N:\\Planet-A\\Data-Master\\')

out_base = 'N:/Planet-A/LUF-Modelling/LUTO2_JZ/biodiversity_data/data'




def process_row(row):
    ds = rxr.open_rasterio(row['path'])
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata)
    dims = {k: [v] for k, v in dict(row).items() if k != 'path'}
    return ds.expand_dims(dims)

# Loop through each model
for idx, group in df.groupby('model'):

    pbar = tqdm(total=len(group))
    para_obj = Parallel(n_jobs=-1, prefer="threads", return_as='generator')
    tasks = (delayed(process_row)(row) for _, row in group.iterrows())
    
    xr_group = []
    for result  in para_obj(tasks):
        xr_group.append(result)
        pbar.update(1)

    xr_group = xr.combine_by_coords(xr_group, fill_value=0)
    xr_group.name = 'data'
    encoding = {'data': {"compression": "gzip", "compression_opts": 9}}
    xr_group.to_netcdf(f'{out_base}/bio_{idx}.nc', encoding=encoding, engine='h5netcdf')
    
    pbar.close()
