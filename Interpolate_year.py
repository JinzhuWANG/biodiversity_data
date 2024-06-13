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
ds_historic['y'] = ds_enssemble['y']
ds_historic['x'] = ds_enssemble['x']
ds = xr.combine_by_coords([ds_historic, ds_enssemble])['data']


# Load an biodiversity map to retrieve the geo-information
with rasterio.open('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif') as src:
    transform = src.transform
    src_arr = src.read(1)
    
# Vectorize the map, each cell will have a unique id as the value    
ds_map = np.arange(src_arr.size).reshape(src_arr.shape)
cells = ({'properties': {'cell_id': v}, 'geometry': s} for s, v in shapes(ds_map, transform=transform))
cell_df = gpd.GeoDataFrame.from_features(list(cells))



# Convert the coordinates to Points
coord_lon_lat = np.load("data/coord_lon_lat_res1.npy")
points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_lon_lat[0], coord_lon_lat[1]))


# Find the cell which has at leat one point in it
joined_gdf = gpd.sjoin(cell_df, points_gdf, how='left').dropna(subset=['index_right'])
joined_gdf = joined_gdf.sort_values('index_right').reset_index(drop=True)
joined_gdf = joined_gdf.rename(columns={'index_right': 'point_id'})
joined_gdf[['cell_id', 'point_id']] = joined_gdf[['cell_id', 'point_id']].astype(np.int64)


# Filter cells from biodiversity map, the cells which have at least one point in it
indices_cell = joined_gdf['cell_id'].unique()
valide_cells = cell_df.query('cell_id in @indices_cell').copy()
valide_cells['masked_cell_id'] = range(valide_cells.shape[0])

mask = np.isin(ds_map, indices_cell)
mask_da = xr.DataArray(mask, dims=['y', 'x'], coords={'y': ds.coords['y'], 'x': ds.coords['x']})
mask_da = mask_da.stack(cell=('y', 'x'))


stacked_data = ds.stack(cell=('y', 'x'))
flattened_data = stacked_data.where(mask_da, drop=True).astype(np.int8)
flattened_data = flattened_data.drop_vars(['cell', 'y', 'x'])
flattened_data['cell'] = range(mask.sum())





# Read data
valide_cells = valide_cells
flattened_data = flattened_data

coord_lon_lat = np.load("data/coord_lon_lat_res10.npy")
points_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_lon_lat[0], coord_lon_lat[1]))

# Find the cell which has at leat one point in it
joined_gdf = gpd.sjoin(valide_cells, points_gdf, how='left').dropna(subset=['index_right'])
joined_gdf = joined_gdf.sort_values('index_right').reset_index(drop=True)
joined_gdf = joined_gdf.rename(columns={'index_right': 'point_id'})
joined_gdf[['cell_id', 'point_id']] = joined_gdf[['cell_id', 'point_id']].astype(np.int64)


# Filter the cells from the biodiversity data
cell_indices = joined_gdf['masked_cell_id'].unique()                    # Unique cell indices
cell_value = flattened_data.sel(cell=cell_indices)
cell_interp = cell_value.interp(
    year=[2010], 
    method='linear', 
    kwargs={'fill_value': 'extrapolate'}).round().astype(np.int8)


# Get the point values
point_indices = joined_gdf['masked_cell_id'].values                     # All cell indices
cell_val = cell_interp.sel(cell=point_indices)


