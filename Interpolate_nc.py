import os

from glob import glob
from tqdm.auto import tqdm
from codes import(combine_future_hist, 
                  coord_to_points, 
                  get_bio_cells, 
                  sjoin_cell_point,
                  mask_cells)


# Get all biodiversity data
bio_nc = glob('data/ssp*.nc')


for nc in tqdm(bio_nc, total=len(bio_nc)):
    
    f_name = os.path.basename(nc)

    # Load the data
    ds = combine_future_hist(nc)
    cell_arr, cell_df = get_bio_cells('data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif')
    points_gdf = coord_to_points("data/coord_lon_lat_res1.npy")


    # Find the cell which has at leat one point in it
    joined_gdf = sjoin_cell_point(cell_df, points_gdf)
    masked_cell = mask_cells(ds, cell_arr, joined_gdf)


    # Save data
    if not os.path.exists('data/bio_valid_cells.geojson'):
        indices_cell = joined_gdf['cell_id'].unique()
        valide_cells = cell_df.query('cell_id in @indices_cell').copy()
        valide_cells['masked_cell_id'] = range(valide_cells.shape[0])
        valide_cells.to_file('data/bio_valid_cells.geojson', driver='GeoJSON')

    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'int8'}}
    masked_cell.to_netcdf(f'data/masked_{f_name}', mode='w', encoding=encoding, engine='h5netcdf')

