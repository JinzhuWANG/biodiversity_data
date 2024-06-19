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

    # Get the 2D mask array where True values indicate the cell indices to keep
    mask = np.isin(cell_arr, indices_cell)
    
    # Flatten the 2D mask array to 1D and keep the x/y coordinates coordinates under the dimension 'cell'
    mask_da = xr.DataArray(mask, dims=['y', 'x'], coords={'y': ds.coords['y'], 'x': ds.coords['x']})
    mask_da = mask_da.stack(cell=('y', 'x'))

    # Flatten the dataset to 1D and keep the xy coordinates under the dimension 'cell'
    stacked_data = ds.stack(cell=('y', 'x'))
    
    # Apply the mask to the flattened dataset
    flattened_data = stacked_data.where(mask_da, drop=True).astype(ds.dtype)
    flattened_data = flattened_data.drop_vars(['cell', 'y', 'x'])
    flattened_data['cell'] = range(mask.sum())
    return flattened_data



def get_id_map_by_upsample_reproject(low_res_map, high_res_map, low_res_crs, low_res_trans):
    """
    Upsamples and reprojects a low-resolution map to match the resolution and 
    coordinate reference system (CRS) of a high-resolution map.
    
    Parameters:
        low_res_map (2D xarray.DataArray): The low-resolution map to upsample and reproject. Should at least has the affine transformation.
        high_res_map (2D xarray.DataArray): The high-resolution map to match the resolution and CRS to. Must have CRS and affine transformation.
        low_res_crs (str): The CRS of the low-resolution map.
        low_res_trans (affine.Affine): The affine transformation of the low-resolution map.
    
    Returns:
        xarray.DataArray: The upsampled and reprojected map with the same CRS and resolution as the high-resolution map.
    """
    low_res_id_map = np.arange(low_res_map.size).reshape(low_res_map.shape)
    low_res_id_map = xr.DataArray(
        low_res_id_map, 
        dims=['y', 'x'], 
        coords={'y': low_res_map.coords['y'], 'x': low_res_map.coords['x']})
    
    low_res_id_map = low_res_id_map.rio.write_crs(low_res_crs)
    low_res_id_map = low_res_id_map.rio.write_transform(low_res_trans)
    low_res_id_map = low_res_id_map.rio.reproject_match(
        high_res_map, 
        Resampling = Resampling.nearest, 
        nodata=low_res_map.size + 1).chunk('auto')
    
    return low_res_id_map
    


def bincount_avg(mask_arr, weight_arr, low_res_xr: xr.DataArray=None):
    """
    Calculate the average of weighted values based on bin counts.

    Parameters:
    - mask_arr (2D, xarray.DataArray): Array containing the mask values. 
    - weight_arr (2D, xarray.DataArray, >0 values are valid): Array containing the weight values.
    - low_res_xr (2D, xarray.DataArray): Low-resolution array containing 
        `y`, `x`, `CRS`, and `transform` to restore the bincount stats.

    Returns:
    - bin_avg (xarray.DataArray): Array containing the average values based on bin counts.
    """
    bin_sum = np.bincount(mask_arr.values.flatten(), weights=weight_arr.values.flatten(), minlength=low_res_xr.size)
    bin_occ = np.bincount(mask_arr.values.flatten(), weights=weight_arr.values.flatten() > 0, minlength=low_res_xr.size)

    # Take values up to the last valid index, because the index of `low_res_xr.size + 1` indicates `NODATA`
    bin_sum = bin_sum[:low_res_xr.size + 1]     
    bin_occ = bin_occ[:low_res_xr.size + 1]     

    # Calculate the average value of each bin
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_avg = (bin_sum / bin_occ).reshape(low_res_xr.shape).astype(np.float32)
        bin_avg = np.nan_to_num(bin_avg)
        
    # Restore the bincount stats to the low-resolution array    
    bin_avg = xr.DataArray(
        bin_avg, 
        dims=low_res_xr.dims, 
        coords=low_res_xr.coords)

    # Expand the dimensions of the bin_avg array to match the original weight_arr
    append_dims = {dim: weight_arr[dim] for dim in weight_arr.dims if dim not in bin_avg.dims}
    bin_avg = bin_avg.expand_dims(append_dims)
    
    # Write the CRS and transform to the output array
    bin_avg = bin_avg.rio.write_crs(low_res_xr.rio.crs)
    bin_avg = bin_avg.rio.write_transform(low_res_xr.rio.transform())
    
    return bin_avg



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
    - map_ (1D, np.ndarray): The map to be matched.
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
    
    bio_id_map = get_id_map_by_upsample_reproject(bio_map, NLUM, NLUM.rio.crs, bio_map.rio.transform())
    map_ = bincount_avg(bio_id_map, map_,  bio_map)
    map_ = map_.where(map_ != map_.rio.nodata, 0)
    return map_


