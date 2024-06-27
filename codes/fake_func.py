'''
These are the functions that are used to mimic the LUTO project.

By importing these functions, we can pretent that we are working inside the LUTO project, 
so that it will be easier to migrate testing codes back to LUTO project.

AUTHOR: Jinzhu WANG
EMAIL: wangjinzhulala@gmail.com
DATE: 27 Jun 2024
'''




import pandas as pd
import numpy as np
import rasterio


def get_coarse2D_map(data, map_:np.ndarray)-> np.ndarray:
    """
    Generate a coarse 2D map based on the input data.

    Args:
        data (Data): The input data used to create the map.
        map_ (np.ndarray): The initial 1D map used to create a 2D map.
        
    Returns:
        np.ndarray: The generated coarse 2D map.

    """
    
    # Fill the 1D map to the 2D map_resfactored.
    map_resfactored = data.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
    np.place(map_resfactored, (map_resfactored != data.MASK_LU_CODE) & (map_resfactored != data.NODATA), map_) 
    return map_resfactored



def upsample_array(data, map_:np.ndarray, factor:int) -> np.ndarray:
    """
    Upsamples the given array based on the provided map and factor.

    Parameters:
    data (object): The input data to derive the original dense_2D shape from NLUM mask.
    map_ (2D, np.ndarray): The map used for upsampling.
    factor (int): The upsampling factor.

    Returns:
    np.ndarray: The upsampled array.
    """
    dense_2D_shape = data.NLUM_MASK.shape
    dense_2D_map = np.repeat(np.repeat(map_, factor, axis=0), factor, axis=1)       # Simply repeate each cell by `factor` times at every row/col direction  
    
    
    # Make sure it has the same shape as the original full-res 2D map
    dense_2D_map = dense_2D_map[0:dense_2D_shape[0], 0:dense_2D_shape[1]]           

    # Pad the array
    pad_height = dense_2D_shape[0] - dense_2D_map.shape[0]
    pad_width = dense_2D_shape[1] - dense_2D_map.shape[1]
    dense_2D_map = np.pad(
        dense_2D_map, 
        pad_width=((0, pad_height), (0, pad_width)), 
        mode='edge')

    
    filler_mask = data.LUMAP_2D != data.MASK_LU_CODE
    dense_2D_map = np.where(filler_mask, dense_2D_map, data.MASK_LU_CODE)           # Apply the LU mask to the dense 2D map.
    dense_2D_map = np.where(data.NLUM_MASK, dense_2D_map, data.NODATA)              # Apply the NLUM mask to the dense 2D map.
    return dense_2D_map





class Data:
    def __init__(self, resfactor:int):
        with rasterio.open('data/NLUM_2010-11_mask.tif') as src:
            self.GEO_META_FULL_RES = src.meta.copy()                            
            self.NLUM_MASK = src.read(1)
            
        self.LUMAP_NO_RESFACTOR = pd.read_hdf("data/dvars/lumap.h5").to_numpy()  
        self.LUMAP_2D = np.load('data/dvars/LUMAP_2D.npy')
        self.LUMAP_2D_RESFACTORED = self.LUMAP_2D[int(resfactor/2)::resfactor, int(resfactor/2)::resfactor] if resfactor > 1 else None
        self.MASK_LU_CODE = -1
        self.NODATA = -9999
        
        self.LANDMANS = ['dry', 'irr']
        self.NON_AGRICULTURAL_LANDUSES = [
            'Environmental Plantings',
            'Riparian Plantings',
            'Sheep Agroforestry',
            'Beef Agroforestry',
            'Carbon Plantings (Block)',
            'Sheep Carbon Plantings (Belt)',
            'Beef Carbon Plantings (Belt)',
            'BECCS']
        self.AGRICULTURAL_LANDUSES = [
            'Apples', 'Beef - modified land', 'Beef - natural land', 'Citrus', 
            'Cotton', 'Dairy - modified land', 'Dairy - natural land', 'Grapes', 
            'Hay', 'Nuts', 'Other non-cereal crops', 'Pears', 'Plantation fruit', 
            'Rice', 'Sheep - modified land', 'Sheep - natural land', 'Stone fruit', 
            'Sugar', 'Summer cereals', 'Summer legumes', 'Summer oilseeds', 
            'Tropical stone fruit', 'Unallocated - modified land', 'Unallocated - natural land', 
            'Vegetables', 'Winter cereals', 'Winter legumes', 'Winter oilseeds']
        
