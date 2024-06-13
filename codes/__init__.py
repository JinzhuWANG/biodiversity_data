import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio

from rasterio.features import shapes


def combine_future_hist(nc_path: str, hist_path: str = 'data/historic_historic.nc'):
    """
    Combines future and historic datasets by coordinates.

    Parameters:
    nc_path (str): The file path of the future dataset.
    hist_path (str): The file path of the historic dataset. Default is 'data/historic_historic.nc'.

    Returns:
    xr.Dataset: The combined dataset.

    """
    ds_enssemble = xr.open_dataset(nc_path, engine='h5netcdf', chunks='auto')['data']
    ds_historic = xr.open_dataset(hist_path, engine='h5netcdf', chunks='auto')['data']
    ds_historic['y'] = ds_enssemble['y']
    ds_historic['x'] = ds_enssemble['x']
    return xr.combine_by_coords([ds_historic, ds_enssemble])['data']




def get_bio_cells(bio_map:str) -> gpd.GeoDataFrame:
    """
    Retrieves the biodiversity cells from a given biodiversity map.

    Parameters:
    bio_map (str): The file path of the biodiversity map.

    Returns:
    tuple: A tuple containing the cell map array and a GeoDataFrame of the cells.
    """

    # Load a biodiversity map to retrieve the geo-information
    with rasterio.open(bio_map) as src:
        transform = src.transform
        src_arr = src.read(1)
        
    # Vectorize the map, each cell will have a unique id as the value    
    cell_map_arr = np.arange(src_arr.size).reshape(src_arr.shape)
    cells = ({'properties': {'cell_id': v}, 'geometry': s} for s, v in shapes(cell_map_arr, transform=transform))
    return cell_map_arr, gpd.GeoDataFrame.from_features(list(cells))



def coord_to_points(coord_path:str, crs:str='epsg:4326') -> gpd.GeoDataFrame:
    """
    Convert coordinate data to a GeoDataFrame of points.

    Parameters:
    coord_path (str): The file path to the coordinate data.
    crs (str): The coordinate reference system (CRS) of the points. Default is 'epsg:4326'.

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the points.

    """
    coord_lon_lat = np.load(coord_path)
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(coord_lon_lat[0], coord_lon_lat[1])).set_crs(crs)



def sjoin_cell_point(cell_df:gpd.GeoDataFrame, points_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially joins a GeoDataFrame of cells with a GeoDataFrame of points.

    Parameters:
    - cell_df (gpd.GeoDataFrame): The GeoDataFrame of cells.
    - points_gdf (gpd.GeoDataFrame): The GeoDataFrame of points.

    Returns:
    - joined_gdf (gpd.GeoDataFrame): The joined GeoDataFrame with the cells and points.

    """
    joined_gdf = gpd.sjoin(cell_df, points_gdf, how='left').dropna(subset=['index_right'])
    joined_gdf = joined_gdf.sort_values('index_right').reset_index(drop=True)
    joined_gdf = joined_gdf.rename(columns={'index_right': 'point_id'})
    joined_gdf[['cell_id', 'point_id']] = joined_gdf[['cell_id', 'point_id']].astype(np.int64)
    return joined_gdf


def mask_cells(ds: xr.Dataset, cell_arr: np.ndarray, joined_gdf: gpd.GeoDataFrame) -> xr.Dataset:
    """
    Masks the cells in a dataset based on a given array of cell indices.

    Parameters:
        ds (xr.Dataset): The input dataset.
        cell_arr (np.ndarray): Array of cell indices.
        joined_gdf (gpd.GeoDataFrame): GeoDataFrame containing cell IDs.

    Returns:
        xr.Dataset: The masked dataset with cells filtered based on the given indices.
    """
    indices_cell = joined_gdf['cell_id'].unique()

    mask = np.isin(cell_arr, indices_cell)
    mask_da = xr.DataArray(mask, dims=['y', 'x'], coords={'y': ds.coords['y'], 'x': ds.coords['x']})
    mask_da = mask_da.stack(cell=('y', 'x'))

    stacked_data = ds.stack(cell=('y', 'x'))
    flattened_data = stacked_data.where(mask_da, drop=True).astype(np.int8)
    flattened_data = flattened_data.drop_vars(['cell', 'y', 'x'])
    flattened_data['cell'] = range(mask.sum())
    return flattened_data