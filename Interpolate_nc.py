import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import Point, Polygon
from rasterio.features import shapes

from tqdm.auto import tqdm
from joblib import Parallel, delayed


# Define parameters
workers = -1

# Load the data
ds_enssemble = xr.open_dataset('data/ssp245_EnviroSuit.nc', engine='h5netcdf', chunks='auto')['data']
ds_historic = xr.open_dataset('data/historic_historic.nc', engine='h5netcdf', chunks='auto')['data']

# Fix the rounding error for x/y coords between the two datasets
ds_historic['y'] = ds_enssemble['y']
ds_historic['x'] = ds_enssemble['x']


# Combine the two datasets
ds = xr.combine_by_coords([ds_historic, ds_enssemble])['data']


# Load an abitrary map from ds, and then vectorize each cell into a polygon
with rasterio.open('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif') as src:
    transform = src.transform
    crs = src.crs
    bounds = src.bounds
    src_arr = src.read(1)
    

ds_map = np.arange(src_arr.size).reshape(src_arr.shape)

results = (
    {'properties': {'cell_id': v}, 'geometry': s}
    for s, v in shapes(ds_map, transform=transform)
)

cell_df = gpd.GeoDataFrame.from_features(list(results))



# Convert the coordinates to Points
coord_lon_lat = np.load("data/coord_lon_lat_res1.npy")
coords_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_lon_lat[0], coord_lon_lat[1]))



# Find the cell which has at leat a point in it
joined_gdf = gpd.sjoin(cell_df, coords_gdf, how='left').dropna(subset=['index_right'])
joined_gdf = joined_gdf.sort_values('index_right').reset_index(drop=True)
indices = joined_gdf['cell_id'].values.astype(np.int32)




def interp_sel(ds, indices):
    da = ds.data.compute()
    da = da.reshape([ds['year'].size, ds['species'].size, -1])
    sel_val = da[..., indices]
    return xr.DataArray(sel_val, 
                        dims=['year', 'species', 'points'],
                        coords={
                            'year': ds['year'],
                            'species': ds['species'],
                            'points': indices}, 
                        )





# Parallel processing to get the interpolated values
species_chunk_num = 1000
species_chunk_idx = np.array_split(np.arange(ds['species'].size), species_chunk_num)
ds_chunks = [ds.isel(species=idx) for idx in species_chunk_idx]


tasks = [delayed(interp_sel)(da_arr, indices) for da_arr in ds_chunks]
para_obj = Parallel(n_jobs=workers, return_as='generator')


val_pts = []
for out in tqdm(para_obj(tasks), total=len(tasks)):
    val_pts.append(out)


# Combine the results
val_pts = xr.combine_by_coords(val_pts)


val_pts_intep = val_pts.interp(
    year=[2011,2012],
    method='linear',
    kwargs={'fill_value': 'extrapolate'}
).round().astype(np.int8)


# Save to nc
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'int8'}} 
val_pts.to_netcdf('data/bio_ensemble_interpolated.nc', mode='w', encoding=encoding, engine='h5netcdf')
