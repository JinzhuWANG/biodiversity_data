
import os
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray as rxr
import sparse

import luto.settings as settings

from itertools import product
from tqdm.auto import tqdm
from joblib import delayed
from rasterio.enums import Resampling
from rasterio.features import shapes
from codes.fake_func import upsample_array, Data, get_coarse2D_map

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES




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
    return ag_dvar_xr.reindex(lu=data.AGRICULTURAL_LANDUSES)


def am_to_xr(data, am_dvar):
    am_dvars = []
    for am in am_dvar.keys():  
        am_dvar_xr = xr.DataArray(
            am_dvar[am], 
            dims=('lm', 'cell', 'lu'),
            coords={
                'lm': data.LANDMANS,
                'cell': np.arange(am_dvar[am].shape[1]),
                'lu': data.AGRICULTURAL_LANDUSES
            }   
        )
        am_dvar_xr = am_dvar_xr.expand_dims({'am':[am]})
        am_dvars.append(am_dvar_xr)
            
    return xr.combine_by_coords(am_dvars).reindex(am=AG_MANAGEMENTS_TO_LAND_USES.keys(), lu=data.AGRICULTURAL_LANDUSES, lm=data.LANDMANS)



def non_ag_to_xr(data, dvar):
    non_ag_dvar_xr = xr.DataArray(
        dvar, 
        dims=('cell', 'lu'),
        coords={
                'cell': np.arange(dvar.shape[0]),
                'lu': data.NON_AGRICULTURAL_LANDUSES}   
    )
    return non_ag_dvar_xr.reindex(lu=data.NON_AGRICULTURAL_LANDUSES)


def get_id_gdf(ds:xr.DataArray, crs:str='epsg:4283') -> gpd.GeoDataFrame:
    """
    Vectorize a biodiversity map to individual cells.

    Parameters:
    ds (xr.DataArray): The biodiversity map as an xarray DataArray. Must be a 2D xarray with `transform` metadata.
    crs (str, optional): The coordinate reference system of the output GeoDataFrame. Defaults to 'epsg:4283' (GDA 1994).

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame representing the unique id of each cell in the input `ds`. E.g., if the ds 
    is of size 100x100 (y*x), then the id_gdf will be 10000 rows, and each row will be a rectagle covering the cell,
    and has a unique value starting from 0 to 9999 in the row-by-row order.
    
    The columns of the GeoDataFrame are:
    - geometry: The rectangle representing the coverage of cells.
    - cell_id: The unique cell index. 
    """

    # Load a biodiversity map template to retrieve the geo-information
    transform = ds.rio.transform()
    src_arr = ds.values
    
    # Vectorize the map, each cell will have a unique id as the value    
    cell_map_arr = np.arange(src_arr.size).reshape(src_arr.shape)
    cells = ({'properties': {'cell_id': int(v)}, 'geometry': s} for s, v in shapes(cell_map_arr, transform=transform))
    return gpd.GeoDataFrame.from_features(list(cells)).set_crs(crs)






def sjoin_cell_point(cell_df:gpd.GeoDataFrame, points_gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatially joins a GeoDataFrame of rectangle cells with a GeoDataFrame of points. Only the rectangle cells that
    have at least one point inside are kept. The rows of `cell_df` will be duplicated if they have multiple points inside.
    E.g., if a cell has 3 points inside, then the cell will be duplicated 3 times in the output GeoDataFrame. However,
    These duplicated cells will have the same `cell_id` but different `point_id`.

    Parameters:
    - cell_df (gpd.GeoDataFrame): The GeoDataFrame of rectangel cells.
    - points_gdf (gpd.GeoDataFrame): The GeoDataFrame of points.

    Returns:
    joined_gdf (gpd.GeoDataFrame): The joined GeoDataFrame with the cells and points. The columns are:
    - geometry: The centroid of the cell.
    - cell_id: The unique cell index.
    - point_id: The unique point index.

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
        - ds (xr.Dataset): The input dataset. An xarray Dataset with `x` and `y` as the controid of each cell.
        - cell_arr (np.ndarray): Array of cell indices for the input `ds`. E.g., if the ds arrary is 100x100, 
          then the id_arr is of the same shape, and each cell has a unique value starting 
          from 0 to 9999 from row-by-row order
        - joined_gdf (gpd.GeoDataFrame): GeoDataFrame of overlapping a map with LUTO_input_coords.
          We can think of this as a spatial join between a map and many points; and each row in the joined_gdf
          represents a cell of at least having one point inside. The columns are:
            - geometry: The centroid of the cell.
            - cell_id: The unique cell index.
            - point_id: The unique point index.

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



def get_id_map_by_upsample_reproject(low_res_map, high_res_map):
    """
    Upsamples and reprojects a low-resolution map to match the resolution and 
    coordinate reference system (CRS) of a high-resolution map.
    
    Parameters:
        low_res_map (2D xarray.DataArray): The low-resolution map to upsample and reproject. Should have CRS and affine transformation.
        high_res_map (2D xarray.DataArray): The high-resolution map to match the resolution and CRS to. Should have CRS and affine transformation.
    
    Returns:
        xarray.DataArray: The upsampled and reprojected map with the same CRS and resolution as the high-resolution map.
    """
    low_res_id_map = np.arange(low_res_map.size).reshape(low_res_map.shape)
    low_res_id_map = xr.DataArray(
        low_res_id_map, 
        dims=['y', 'x'], 
        coords={'y': low_res_map.coords['y'], 'x': low_res_map.coords['x']})
    
    low_res_id_map = low_res_id_map.rio.write_crs(low_res_map.rio.crs)
    low_res_id_map = low_res_id_map.rio.write_transform(low_res_map.rio.transform())
    low_res_id_map = low_res_id_map.rio.reproject_match(
        high_res_map, 
        Resampling = Resampling.nearest, 
        nodata=low_res_map.size + 1).chunk('auto')
    
    return low_res_id_map
    


def bincount_avg(bin_arr, weight_arr, low_res_xr: xr.DataArray=None):
    """
    Calculate the average of weighted values based on bin counts.

    Parameters:
    - bin_arr (2D, xarray.DataArray): Array containing the mask values. 
    - weight_arr (2D, xarray.DataArray, >0 values are valid): Array containing the weight values.
    - low_res_xr (2D, xarray.DataArray): Low-resolution array containing 
        `y`, `x`, `CRS`, and `transform` to restore the bincount stats.

    Returns:
    - bin_avg (xarray.DataArray): Array containing the average values based on bin counts.
    """
    # Get the valide cells
    valid_mask = weight_arr > 0
    
    # Flatten arries
    bin_flatten = bin_arr.values[valid_mask]
    weights_flatten = weight_arr.values[valid_mask]
    
    bin_occ = np.bincount(bin_flatten, minlength=low_res_xr.size)
    bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=low_res_xr.size)
    
    # Take values up to the last valid index, because the index of `low_res_xr.size + 1` indicates `NODATA`
    bin_sum = bin_sum[:low_res_xr.size + 1]     
    bin_occ = bin_occ[:low_res_xr.size + 1]     

    # Calculate the average value of each bin, ignoring division by zero (which will be nan)
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
    bio_id_path:str=f'{settings.INPUT_DIR}/bio_id_map.nc',
    lumap_tempelate:str=f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif', 
    biomap_tempelate:str=f'{settings.INPUT_DIR}/bio_mask.nc')-> xr.DataArray:
    """
    Matches the lumap and biomap data based on the given parameters.

    Args:
        data (Data): The data object.
        map_ (np.ndarray): The map array.
        res_factor (int): The resolution factor.
        bio_id_path (str, optional): The path to the bio id map. Defaults to f'{settings.INPUT_DIR}/bio_id_map.nc'.
        lumap_tempelate (str, optional): The path to the lumap template. Defaults to f'{settings.INPUT_DIR}/NLUM_2010-11_mask.tif'.
        biomap_tempelate (str, optional): The path to the biomap template. Defaults to f'{settings.INPUT_DIR}/bio_mask.nc'.

    Returns:
        xr.DataArray: The matched map array.
    """
    # Read lumap, bio_map, and bio_id_map
    NLUM = rxr.open_rasterio(lumap_tempelate, chunks='auto').squeeze('band').drop_vars('band')
    bio_mask_ds = xr.open_dataset(f'{settings.INPUT_DIR}/bio_mask.nc', decode_coords="all")
    bio_map = bio_mask_ds['data']
    bio_map['spatial_ref'] = bio_mask_ds['spatial_ref']
    bio_id_map = xr.open_dataset(bio_id_path, chunks='auto')['data']
        
    if res_factor > 1:   
        map_ = get_coarse2D_map(data, map_)
        map_ = upsample_array(data, map_, res_factor)
    else:
        empty_map = np.full(data.NLUM_MASK.shape, data.NODATA).astype(np.float32) 
        np.place(empty_map, data.NLUM_MASK, data.LUMAP_NO_RESFACTOR) 
        np.place(empty_map, empty_map >=0, map_.data.todense())
        map_ = empty_map
        
    map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': NLUM['y'], 'x': NLUM['x']})
    map_ = map_.where(map_>=0, 0)
    map_ = map_.rio.write_crs(NLUM.rio.crs)
    map_ = map_.rio.write_transform(NLUM.rio.transform())  
    map_ = bincount_avg(bio_id_map, map_,  bio_map)
    map_ = map_.where(map_ != map_.rio.nodata, 0)
    return map_


def ag_dvar_to_bio_map(data, ag_dvar, res_factor, para_obj):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - para_obj (object): The parallel processing object.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(ag_dvar, lm, lu, res_factor):
        map_ = ag_dvar.sel(lm=lm, lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'lm': [lm], 'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(ag_dvar, lm, lu, res_factor) 
             for lm,lu in product(ag_dvar['lm'].values, ag_dvar['lu'].values)]
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr


def am_dvar_to_bio_map(data, am_dvar, res_factor, para_obj):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - am_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - para_obj (object): The parallel processing object.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(am_dvar, am, lm, lu, res_factor):
        map_ = am_dvar.sel(lm=lm, lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'am':[am], 'lm': [lm], 'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(am_dvar, am, lm, lu, res_factor) 
            for am,lm,lu in product(am_dvar['am'].values, am_dvar['lm'].values, am_dvar['lu'].values)]
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr



def non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, para_obj):
    """
    Reprojects and matches agricultural land cover variables to biodiversity maps.

    Parameters:
    - data (Data object): The Data class object of LUTO.
    - non_ag_dvar (xarray.Dataset): The agricultural land cover variables.
    - res_factor (int): The resolution factor for matching.
    - para_obj (object): The parallel processing object.

    Returns:
    - xarray.Dataset: The combined dataset of reprojected and matched variables.
    """

    # wrapper function for parallel processing
    def reproject_match_dvar(non_ag_dvar, lu, res_factor):
        map_ = non_ag_dvar.sel(lu=lu)
        map_ = match_lumap_biomap(data, map_, res_factor)
        map_ = map_.expand_dims({'lu': [lu]})
        return map_

    tasks = [delayed(reproject_match_dvar)(non_ag_dvar, lu, res_factor) 
            for lu in non_ag_dvar['lu'].values]
    out_arr = xr.combine_by_coords([i for i in para_obj(tasks)])
    # Convert to sparse array to save memory
    out_arr.values = sparse.COO.from_numpy(out_arr.values)
    
    return out_arr


# Calculate the biodiversity condition
def calc_bio_hist_sum(bio_nc_path:str):
    bio_xr_raw = xr.open_dataset(bio_nc_path, chunks='auto')['data']
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint64'}}
    bio_xr_mask = xr.open_dataset('data/bio_mask.nc', chunks='auto')['data'].astype(np.bool_)
    
    # Get the sum of all historic cells, !!! important !!! need to mask out the cells that are not in the bio_mask 
    if os.path.exists('data/bio_xr_hist_sum_species.nc'):
        bio_xr_hist_sum_species = xr.open_dataarray(f'data/bio_xr_hist_sum_species.nc', chunks='auto') 
    else:
        bio_xr_hist_sum_species = (bio_xr_raw.sel(year=1990).astype('uint64') * bio_xr_mask).sum(['x', 'y']).compute()
        bio_xr_hist_sum_species.to_netcdf('data/bio_xr_hist_sum_species.nc', mode='w', encoding=encoding, engine='h5netcdf')   
    return bio_xr_hist_sum_species


# Calculate the biodiversity contribution of each species
def calc_bio_score_species(bio_nc_path:str, bio_xr_hist_sum_species: xr.DataArray):
    """
    Calculate the contribution of each cell to the total biodiversity score for a given species.

    Parameters:
    bio_nc_path (str): The file path to the biodiversity data in NetCDF format.
    bio_xr_hist_sum_species (xr.DataArray): The historical sum of biodiversity scores for the species.

    Returns:
    xr.DataArray: The contribution of each cell to the total biodiversity score as a percentage.
    """
    bio_xr_raw = xr.open_dataset(bio_nc_path, chunks='auto')['data']
    # Calculate the contribution of each cell (%) to the total biodiversity
    bio_xr_contribution_species = (bio_xr_raw / bio_xr_hist_sum_species).astype(np.float32) * 100   
    return bio_xr_contribution_species


# Calculate the biodiversity contribution of each group (amphibians, mammals, etc.)
def calc_bio_score_group(bio_nc_path:str, bio_xr_hist_sum_species: xr.DataArray):
    """
    Calculate the contribution of each cell to the total biodiversity score for a given group (ampibians, mammals, etc.).

    Parameters:
    - bio_nc_path (str): The file path to the biodiversity data in NetCDF format.
    - bio_xr_hist_sum_species (xr.DataArray): The historical sum of bio-score for each group.

    Returns:
    - bio_xr_contribution_group (xr.DataArray): The biodiversity score for each group.

    """
    bio_contribution_species = calc_bio_score_species(bio_nc_path, bio_xr_hist_sum_species)
    groups = list(set(bio_contribution_species['group'].values))
    bio_contribution_group = bio_contribution_species.groupby('group').mean(dim='species').astype(np.float32)
    return bio_contribution_group


# Helper functions to interpolate biodiversity scores to given years
def interp_by_year(ds, year:list[int]):
    return ds.interp(year=year, method='linear', kwargs={'fill_value': 'extrapolate'}).astype(np.float32).compute()

def interp_bio_species_to_shards(bio_contribution_species, interp_year, max_workers=settings.THREADS):
    """
    Interpolates biodiversity data for different species to shards.

    Parameters:
    - bio_contribution_species: xarray.Dataset
        The biodiversity data for different species.
    - interp_year: list
        The years for which the data needs to be interpolated.
    - max_workers: int, optional
        The maximum number of workers to use for parallel processing. Defaults to settings.THREADS.

    Returns:
    - list
        A list of delayed objects representing the interpolation of biodiversity data for each species at interp_year.
    """
    chunks_species = np.array_split(range(bio_contribution_species['species'].size), max_workers)
    bio_xr_chunks = [bio_contribution_species.isel(species=idx) for idx in chunks_species]
    return [delayed(interp_by_year)(chunk, [yr]) for chunk in bio_xr_chunks for yr in interp_year]


def interp_bio_group_to_shards(bio_contribution_group, interp_year):
    """
    Interpolates the biodiversity contribution group to shards for the given interpolation years.

    Parameters:
    - bio_contribution_group (list): List of biodiversity contribution values for a group.
    - interp_year (list): List of years for interpolation.

    Returns:
    - List of delayed objects representing the interpolation of biodiversity data for each group at interp_year.
    """
    return [delayed(interp_by_year)(bio_contribution_group, [year]) for year in interp_year]


# Helper functions to calculate the biodiversity contribution scores
def xr_to_df(shard, dvar_ag, dvar_am, dvar_non_ag):
    """
    Convert an xarray DataArray to a pandas DataFrame with biodiversity contribution scores.

    Parameters:
    - shard: An xarray DataArray or a tuple of delayed function and arguments.
    - dvar_ag: An xarray DataArray representing the biodiversity contribution scores for agriculture land use.
    - dvar_am: An xarray DataArray representing the biodiversity contribution scores for amenity land use.
    - dvar_non_ag: An xarray DataArray representing the biodiversity contribution scores for non-agriculture land use.

    Returns:
    - A pandas DataFrame containing the biodiversity contribution scores for different land use types.

    Raises:
    - ValueError: If the shard parameter is not a valid type (either tuple or xarray DataArray).
    """

    # Get the values to avoid duplicate computation
    if isinstance(shard, tuple):            # This means shard is a tuple of delayed function and arguments
        f, args = shard[0], shard[1]
        shard_xr = f(*args)
    elif isinstance(shard, xr.DataArray):   # This means shard is an xr.DataArray
        shard_xr = shard.compute()
    else:
        raise ValueError('Invalid shard type! Should be either tuple (delayed(function), args) or xr.DataArray.')

    # Calculate the biodiversity contribution scores
    tmp_ag = (shard_xr * dvar_ag).compute().sum(['x', 'y'])
    tmp_am = (shard_xr * dvar_am).compute().sum(['x', 'y'])
    tmp_non_ag = (shard_xr * dvar_non_ag).compute().sum(['x', 'y'])
    
    # Convert to dense array
    tmp_ag.data = tmp_ag.data.todense()
    tmp_am.data = tmp_am.data.todense()
    tmp_non_ag.data = tmp_non_ag.data.todense()
    
    # Convert to dataframe
    tmp_ag_df = tmp_ag.to_dataframe(name='contribution_%').reset_index()
    tmp_am_df = tmp_am.to_dataframe(name='contribution_%').reset_index()
    tmp_non_ag_df = tmp_non_ag.to_dataframe(name='contribution_%').reset_index()
    
    # Add land use type
    tmp_ag_df['lu_type'] = 'ag'
    tmp_am_df['lu_type'] = 'am'
    tmp_non_ag_df['lu_type'] = 'non_ag'
    
    return pd.concat([tmp_ag_df, tmp_am_df, tmp_non_ag_df], ignore_index=True)


def calc_bio_score_by_yr(ag_dvar, am_dvar, non_ag_dvar, bio_shards, para_obj):
    """
    Calculate biodiversity score by year.

    Parameters:
    - ag_dvar: The agricultural variable.
    - am_dvar: The agricultural management variable.
    - non_ag_dvar: The non-agricultural variable.
    - bio_shards: The biodiversity shards.
    - para_obj: The parallelization object.

    Returns:
    - The calculated biodiversity score by year, excluding the 'spatial_ref' column.
    """
    
    tasks = [delayed(xr_to_df)(bio_score, ag_dvar, am_dvar, non_ag_dvar) for bio_score in bio_shards]
    out = pd.concat([out for out in tqdm(para_obj(tasks), total=len(tasks))], ignore_index=True)
    return out.drop(columns=['spatial_ref'])



