import os
import numpy as np
import xarray as xr
import geopandas as gpd
import luto.settings as settings
from codes import get_id_gdf, sjoin_cell_point




# Read in lon/lat points of resfactor 1; Pretend they are loaded from LUTO `Data` object
'''The `coord_x` and `coord_y` are the lon/lat points of the LUTO input data, 
which were taken from the `Data.COORD_LON_LAT`.'''
luto_mask = np.load('data/coord/LUTO_MASK_res1.npy')
coord_x = np.load('data/coord/coord_res1_x.npy')
coord_y = np.load('data/coord/coord_res1_y.npy')
coord_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_x, coord_y)).set_crs('epsg:4283')


# Read the template bio file
bio_template_ds = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', chunks='auto')
bio_template = bio_template_ds['data']
bio_template['spatial_ref'] = bio_template_ds['spatial_ref']


# Get the id map/df; 
'''E.g., if the map is size 100x100, then the id_arr is of the same shape, 
and each cell has a unique value starting from 0 to 9999 in row-by-row order'''
id_gdf = get_id_gdf(bio_template)  
id_gdf['area_ha'] = id_gdf['geometry'].copy().to_crs('EPSG:3577').area / 10000  # EPSG:3577 is the Australian Albers projection
joined_gdf = sjoin_cell_point(id_gdf, coord_gdf)


# Save data    
if not os.path.exists(f'{settings.INPUT_DIR}/bio_valid_cells.geojson'):
    uniqu_cell_id = joined_gdf['cell_id'].unique()                       # Unique cell indices, each index corresponds to a cell that at least one point falls into
    valide_cells = id_gdf.query('cell_id in @uniqu_cell_id').copy()      # The cells that have at least one point falling into is valid
    valide_cells['area_ha'] = id_gdf['area_ha'].iloc[uniqu_cell_id]      # Add the area_ha column (ha)
    valide_cells.to_file(f'{settings.INPUT_DIR}/bio_valid_cells.geojson', driver='GeoJSON')
    
    



