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
excl_models = ['GCM-Ensembles']

df = df.query('model not in @excl_models')




def process_row(row, dims=None):
    ds = rxr.open_rasterio(row['path'])
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata)
    dims = {k: [v] for k, v in dict(row).items() if k != 'path'} if dims is None else dims
    return ds.expand_dims(dims)



# Loop through each model
for idx, group in df.groupby('model'):
    
    # Get the output path and dims
    out_path = f'{out_base}/bio_{idx}.nc'
    cols = [i for i in group.columns if i != 'path']
    dims = {k: group[k].unique().tolist() for k in cols }
    
    # Get the chunk size
    row = group.iloc[0]
    row_xr = process_row(row, dims)
    y_idx, x_idx = row_xr.dims.index('y'), row_xr.dims.index('x')
    chunk_size = [1] * row_xr.ndim
    chunk_size[y_idx], chunk_size[x_idx] = 128, 128
    
    # Create the encoding for writing
    encoding = {'data': {"compression": "gzip", "compression_opts": 9, "chunksizes": chunk_size}}
    
    # Multi-threading to read TIF and expand dims
    para_obj = Parallel(n_jobs=-1, prefer="threads", return_as='generator')
    tasks = (delayed(process_row)(row) for _, row in group.iterrows())
    pbar = tqdm(total=len(group))
    
    xr_chunk = []
    for result  in para_obj(tasks):
        xr_chunk.append(result)
        pbar.update(1)

    # Combine the data and write to nc
    xr_chunk = xr.combine_by_coords(xr_chunk, fill_value=0)
    xr_chunk.name = 'data'
    xr_chunk.to_netcdf(out_path, mode='w', encoding=encoding, engine='h5netcdf')
        
    pbar.close()
