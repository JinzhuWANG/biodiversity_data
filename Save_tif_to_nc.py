from codes.helper_func import replace_with_nearest
import rioxarray as rxr
import xarray as xr
import pandas as pd




# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')


# Loop through each model
for idx, group in df.groupby('model'):break

xr_group = []
for _, row in group.iterrows():
    ds = rxr.open_rasterio(row['path'])
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata)
    ds.attrs = {}
    
    dims = {k: [v] for k, v in dict(row).items() if k != 'path'}
    ds = ds.expand_dims(dims)
    xr_group.append(ds)


xr_group = xr.combine_by_coords(xr_group, combine_attrs='override')
xr_group.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9}}
xr_group.to_netcdf(f'out_raster/{idx}.nc', encoding=encoding, engine='h5netcdf')


f = xr.open_dataset(f'out_raster/{idx}.nc')







