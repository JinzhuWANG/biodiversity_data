import os
import re
import numpy as np
import pandas as pd
from osgeo import gdal
import rasterio
import affine

from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT

from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


def replace_with_nearest(map_: np.ndarray, filler: int) -> np.ndarray:
    """
    Replaces invalid values in the input array with the nearest non-filler values.

    Parameters:
        map_ (np.ndarray, 2D): The input array.
        filler (int): The value to be considered as invalid.

    Returns:
        np.ndarray (2D): The array with invalid values replaced by the nearest non-invalid values.
    """
    # Create a mask for invalid values
    mask = (map_ == filler)
    # Perform distance transform on the mask
    _, nearest_indices = distance_transform_edt(mask, return_indices=True)
    # Replace the invalid values with the nearest non-invalid values
    map_[mask] = map_[tuple(nearest_indices[:, mask])]
    
    return map_


def get_warp_opt(in_path:str, resample:Resampling=Resampling.bilinear) -> dict:
    """
    Get the warp options for a given input raster file.

    Parameters:
    - in_path (str): Path to the input raster file.
    - resample (Resampling): Resampling method to be used during warping. Default is Resampling.bilinear.

    Returns:
    - dict: A dictionary containing the warp options:
        - 'resampling': The resampling method.
        - 'crs': The coordinate reference system of the input raster.
        - 'transform': The affine transformation matrix for the output raster.
        - 'height': The height of the input raster.
        - 'width': The width of the input raster.
    """
    with rasterio.open(in_path) as ref:
        left, bottom, right, top = ref.bounds
        xres = (right - left) / ref.width
        yres = (top - bottom) / ref.height
        dst_transform = affine.Affine(xres, 0.0, left, 0.0, -yres, top)
        return {
            'resampling': resample,
            'crs': ref.crs,
            'transform': dst_transform,
            'height': ref.height,
            'width': ref.width
        }



def warp_raster(in_path:str=None, ref_path:str=None) -> None:
    """
    Warps a raster to match the spatial reference and resolution of a reference raster.

    Args:
        in_path (str): Path to the input raster file.
        ref_path (str): Path to the reference raster file.

    Returns:
        numpy.ndarray: The warped raster data as a NumPy array with data type 'int8'.
    """
    
    # Get warp options, and reference raster metadata
    warp_option = get_warp_opt(ref_path)
    with rasterio.open(ref_path) as ref:
        ref_mask = (ref.read(1) != -1) & (ref.read(1) != ref.nodata)

    # Fill the nodata of input
    with rasterio.open(in_path) as src:
        data = src.read(1)
        data = replace_with_nearest(data, src.nodata)  
        src_meta = src.meta.copy()
    
    # Write filled data to memory 
    memfile = MemoryFile()   
    with memfile.open(**src_meta) as mem_f:
        mem_f.write(data,1)

    # Reproject mem_f to match ref_mask 
    with memfile.open() as mem_f, WarpedVRT(mem_f, **warp_option) as vrt:
        data = vrt.read(1)
        data = data[np.nonzero(ref_mask)]
    
    return data.astype('int8')


def find_str(row):
    """
    Extracts relevant information from the given row's path and returns it as a list.

    Args:
        row (pandas.Series): A pandas Series object representing a row of data.
    Returns:
        list: A list containing the extracted information from the row's path.
    Raises:
        IndexError: If the regular expression fails to find a match for the year.
    """
    
    
    reg_year = re.compile('_(\d{4})_').findall(row['path'])[0]
    
    if int(reg_year) < 2010:
        return ['historic', 'historic', reg_year, 'historic']
    
    reg_model = re.compile(f'{row["species"]}_(.*)_ssp').findall(row['path'])[0]
    reg_ssp = re.compile('_(ssp\d*)_').findall(row['path'])[0]
    reg_mode = re.compile('km_(.*).tif').findall(row['path'])[0]
    return [reg_model, reg_ssp, int(reg_year), reg_mode]



def get_all_path(root_dir:str, save_path:str='data/all_suitability_tifs.csv'):
    """
    Retrieves the paths of all TIFF files in the specified root directory and saves them to a CSV file.

    Parameters:
    - root_dir (str): The root directory to search for TIFF files.
    - save_path (str): The path to save the CSV file. Default is 'data/all_suitability_tifs.csv'.

    Returns:
    None
    """
    records = []
    for dirpath, _, filenames in tqdm(os.walk(root_dir)):
        for f in filenames:
            if f.endswith('.tif'):
                group, species = os.path.normpath(dirpath).split('\\')[-2:]
                records.append({'group':group, 'species':species, 'path':os.path.join(dirpath, f)})
    
    # Convert all records to df            
    df = pd.DataFrame(records)            
    df[['model', 'ssp', 'year', 'mode']] = df.apply(lambda x: pd.Series(find_str(x)), axis=1) 
    df.to_csv(save_path, index=False)






















