import xarray as xr
import numpy as np
import geopandas as gpd

from tqdm.auto import tqdm
from joblib import Parallel, delayed
from codes import coord_to_points, sjoin_cell_point



# Define parameters
workers = 50
para_obj = Parallel(n_jobs=workers, return_as='generator')
interp_year = [2040]


# Helper function to applied to each chunk
def interp_by_year(ds, year):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).round().astype(np.int8).compute()

def select_points(ds):
    return ds.sel(cell=point_indices)



# Read data
bio_path = 'data/bio_nc_raw/masked_ssp245_EnviroSuit.nc' # ['data/bio_nc_raw/masked_ssp245_EnviroSuit.nc', 'data/bio_nc_raw/masked_ssp245_EnviroConditionRatio.nc']
valide_cells_xr = xr.open_dataset(bio_path, chunks='auto')['data']
valide_cells_df = gpd.read_file('data/bio_valid_cells.geojson').set_crs('EPSG:4283', allow_override=True) # Reading from geojson loses the crs, so we need to set it again
points_gdf = coord_to_points("data/coord_lon_lat_res1.npy")

# Split the data into chunks at the species level
chunks_idx = np.array_split(range(valide_cells_xr['species'].size), workers)

# Spatial join the cells and the points
joined_gdf = sjoin_cell_point(valide_cells_df, points_gdf)
joined_gdf = joined_gdf.sort_values('point_id')


# Filter the cells from the biodiversity data
cell_indices = joined_gdf['masked_cell_id'].unique()                    # Unique cell indices
cell_value = valide_cells_xr.sel(cell=cell_indices)
cell_chunks = [cell_value.isel(species=idx) for idx in chunks_idx]
cell_tasks = [delayed(interp_by_year)(chunk, interp_year) for chunk in cell_chunks]
cell_interp = xr.combine_by_coords([out for out in tqdm(para_obj(cell_tasks), total=len(cell_tasks))])['data']



# Get the point values
point_indices = joined_gdf['masked_cell_id'].values                     # All cell indices
if len(point_indices) != len(cell_indices):
    chunks_points = [cell_interp.isel(species=idx) for idx in chunks_idx]
    points_tasks = [delayed(select_points)(chunk) for chunk in chunks_points]
    cell_interp = xr.combine_by_coords([out for out in tqdm(para_obj(points_tasks), total=len(points_tasks))])









