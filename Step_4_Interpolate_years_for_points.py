import xarray as xr
import numpy as np
import geopandas as gpd

from joblib import Parallel
from codes import calc_bio_hist_sum, calc_bio_score_species, sjoin_cell_point

import luto.settings as settings


# Define parameters
workers = 50
para_obj = Parallel(n_jobs=workers, return_as='generator')
interp_year = 2040



# Get biodiversity contribution data
if settings.BIO_CALC_LEVEL == 'group':
    '''Biodiversity contribution at `group` level is precalculated for all years of 2010-2100'''
    bio_contribution = xr.open_dataset(f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_Condition_group.nc', chunks='auto')['data']
    bio_score_all_species_mean = bio_contribution.mean('group').expand_dims({'group': ['all_species']})   # Calculate the mean score of all species
    bio_contribution = xr.combine_by_coords([bio_contribution, bio_score_all_species_mean])['data']       # Combine the mean score with the original score
    bio_contribution = bio_contribution.sel(year=[interp_year])                                           # Select the year of interest
elif settings.BIO_CALC_LEVEL == 'species':
    '''Biodiversity contribution at `species` level is NOT precalculated, and only include years [1990, 2030, 2050, 2070, 2090]'''
    bio_raw_path = f'{settings.INPUT_DIR}/bio_ssp{settings.SSP}_EnviroSuit.nc'
    bio_his_score_sum = calc_bio_hist_sum(bio_raw_path)
    bio_contribution = calc_bio_score_species(bio_raw_path, bio_his_score_sum)
    # The bio_contribution at `species` level need to be interpolated to the target year first
    bio_contribution = bio_contribution.interp(
        year=[interp_year], 
        method='linear', 
        kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()



# Read coordinates of the input data from LUTO
coord_x = np.load('data/coord/coord_res1_x.npy')
coord_y = np.load('data/coord/coord_res1_y.npy')
coord_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_x, coord_y)).set_crs('epsg:4283')



# Read the masked id_gdf for biodiversity data
'''The `masked` mean that the cells of the biodiversity map that do not intersect with LUTO input coordinates are removed.'''
masked_id_gdf = gpd.read_file(f'{settings.INPUT_DIR}/bio_valid_cells.geojson').set_crs('EPSG:4283', allow_override=True) # Reading from geojson loses the crs, so we need to set it again


# Spatial join the cells and the points
'''The spatial join between the bio_map cells and the LUTO input coordinates creates links between the two datasets.
The resulted GeoDataframe include index for bio_map (cell_id) and LUTO input coordinates (point_id), So that we can 
easily get the bio_map values corresponding to the LUTO input coordinates.'''
joined_gdf = sjoin_cell_point(masked_id_gdf, coord_gdf)     # `cell_id` is the index of the bio_map, `point_id` is the index of the LUTO input coordinates
joined_gdf = joined_gdf.sort_values('point_id')             # Sort the joined_gdf by `point_id`


# Get the point values
'''The `bio_input_indices` is the index of the bio_map that corresponds to the LUTO input coordinates.'''
bio_input_indices = joined_gdf['cell_id'].values 
# Stack the bio_map to 1D array, so we can select values based on the 1D index                         
bio_contribution_sel = bio_contribution.stack(cell=('y', 'x'))    
# Select the bio_map values through the 1D index (note the isel method)          
bio_contribution_sel = bio_contribution_sel.sel(year=[interp_year]).isel(cell=bio_input_indices) 




if __name__ == '__main__':
    # Sanity check: Plot the selected cells
    cell_indices = joined_gdf['cell_id'].unique()                           # Unique cell indices
    bio_contribution_sel = bio_contribution.stack(cell=('y', 'x'))
    bio_contribution_sel = bio_contribution_sel.isel(cell=cell_indices)
    bio_contribution_sel = bio_contribution_sel.unstack('cell')
    bio_contribution_sel = bio_contribution_sel.reindex_like(bio_contribution_sel)








