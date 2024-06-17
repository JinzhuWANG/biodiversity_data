import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
import rasterio

from rasterio.enums import Resampling
from rasterio.features import shapes
from codes.fake_func import upsample_array, Data, get_coarse2D_map




def ag_to_xr(data, dvar):
    ag_dvar_xr = xr.DataArray(
        dvar, 
        dims=('lm', 'cell', 'lu'),
        coords={
            'lm': data.LANDMANS,
            'cell': np.arange(dvar.shape[1]),
            'lu': data.AGRICULTURAL_LANDUSES
        }   
    )
    return ag_dvar_xr


def am_to_xr(data, am, dvar):
    am_dvar_xr = xr.DataArray(
        dvar, 
        dims=('lm', 'cell', 'lu'),
        coords={
            'lm': data.LANDMANS,
            'cell': np.arange(dvar.shape[1]),
            'lu': data.AGRICULTURAL_LANDUSES
        }   
    )
    return am_dvar_xr.expand_dims({'am': [am]})

def non_ag_to_xr(data, dvar):
    non_ag_dvar_xr = xr.DataArray(
        dvar, 
        dims=('cell', 'lu'),
        coords={
                'cell': np.arange(dvar.shape[0]),
                'lu': data.NON_AGRICULTURAL_LANDUSES}   
    )
    return non_ag_dvar_xr


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




def get_bio_cells(bio_map:str, crs:str='epsg:4328') -> gpd.GeoDataFrame:
    """
    Vectorized a bio_map to individual cells.

    Parameters:
    bio_map (str): The file path of the biodiversity map.
    crs (str, optional): The coordinate reference system of the output GeoDataFrame. Defaults to 'epsg:4328' (GDA 1994).

    Returns:
    tuple: A tuple containing the cell map array and its vectorized GeoDataFrame of each cell.
    """

    # Load a biodiversity map to retrieve the geo-information
    with rasterio.open(bio_map) as src:
        transform = src.transform
        src_arr = src.read(1)
        
    # Vectorize the map, each cell will have a unique id as the value    
    cell_map_arr = np.arange(src_arr.size).reshape(src_arr.shape)
    cells = ({'properties': {'cell_id': v}, 'geometry': s} for s, v in shapes(cell_map_arr, transform=transform))
    return cell_map_arr, gpd.GeoDataFrame.from_features(list(cells)).set_crs(crs)



def coord_to_points(coord_path:str, crs:str='epsg:4328') -> gpd.GeoDataFrame:
    """
    Convert coordinate data to a GeoDataFrame of points.

    Parameters:
    coord_path (str): The file path to the coordinate data.
    crs (str): The coordinate reference system (CRS) of the points. Default is 'epsg:4328' (GDA 1994).

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


def match_lumap_biomap(
    data:Data, 
    map_:np.ndarray, 
    res_factor:int, 
    lumap_tempelate:str='data/NLUM_2010-11_mask.tif', 
    biomap_tempelate:str='data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif')-> xr.DataArray:
    """
    Matches the map_ to the same projection and resolution as of biomap templates.
    
    If the map_ is not in the same resolution as the biomap, it will be upsampled to (1km x 1km) first.
    
    The resampling method is set to average, because the map_ is assumed to be the decision variables (float), 
    representing the percentage of a given land-use within the cell.

    Parameters:
    - data (Data): The data object containing necessary information.
    - map_ (np.ndarray): The map to be matched.
    - res_factor (int): The resolution factor.
    - lumap_tempelate (str): The path to the lumap template file. Default is 'data/NLUM_2010-11_mask.tif'.
    - biomap_tempelate (str): The path to the biomap template file. Default is 'data/Arenophryne_rotunda_BCC.CSM2.MR_ssp126_2030_AUS_5km_EnviroSuit.tif'.

    Returns:
    - xr.DataArray: The matched map.

    """
    
    NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_map = rxr.open_rasterio(biomap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_map = bio_map.rio.write_crs(NLUM.rio.crs)
    
    if res_factor > 1:   
        map_ = get_coarse2D_map(data, map_)
        map_ = upsample_array(data, map_, res_factor)
    else:
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_)
        map_ = empty_map
        
    map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': NLUM['y'], 'x': NLUM['x']})
    map_ = map_.where(map_>=0, 0)
    map_ = map_.rio.write_crs(NLUM.rio.crs)
    map_ = map_.rio.write_transform(NLUM.rio.transform())  
    map_ = map_.rio.reproject_match(bio_map, Resampling = Resampling.average, nodata=-1)
    map_ = map_.where(map_ != map_.rio.nodata, 0)
    return map_