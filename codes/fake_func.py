'''
These are the functions that are used to mimic the LUTO project.

By importing these functions, we can pretent that we are working inside the LUTO project, 
so that it will be easier to migrate testing codes back to LUTO project.

AUTHOR: Jinzhu WANG
EMAIL: wangjinzhulala@gmail.com
DATE: 27 Jun 2024
'''

import os
import pandas as pd
import numpy as np
import rasterio
import xarray as xr
import luto.settings as settings

from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

from itertools import product
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class Data:
    def __init__(self, resfactor:int):
        
        self.ag_dvars_2D_reproj_match = {}
        self.am_dvars_2D_reproj_match = {}
        self.non_ag_dvars_2D_reproj_match = {}
        
        
        settings.RESFACTOR = resfactor

        
        # Setup output containers
        self.ag_dvars_2D_reproj_match = {}
        self.non_ag_dvars_2D_reproj_match = {}
        self.ag_man_dvars_2D_reproj_match = {}
        
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

        
        with rasterio.open('input/NLUM_2010-11_mask.tif') as src:
            self.GEO_META_FULL_RES = src.meta.copy()                            
            self.NLUM_MASK = src.read(1)
            
        self.LUMAP_NO_RESFACTOR = pd.read_hdf("data/dvars/lumap.h5").to_numpy()  
        self.LUMAP_2D = np.load('data/dvars/LUMAP_2D.npy')
        self.LUMAP_2D_RESFACTORED = self.LUMAP_2D[int(resfactor/2)::resfactor, int(resfactor/2)::resfactor] if resfactor > 1 else None
        self.MASK_LU_CODE = -1
        self.NODATA = -9999
        
        dvar_path = 'data/dvars/res5_fullrun'
        years = [int(i[-4:]) for i in os.listdir(dvar_path) if 'out_' in i]
        
        self.ag_dvars = {year:np.load(f'{dvar_path}/out_{year}/ag_X_mrj_{year}.npy') for year in years}       
        self.non_ag_dvars = {year:np.load(f'{dvar_path}/out_{year}/non_ag_X_rk_{year}.npy') for year in years}       
        self.ag_man_dvars = {year: {k: np.load(f'{dvar_path}/out_{year}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{year}.npy') for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}  
                            for year in years}   
        


        
    # Functions to add reprojected dvars to the output containers
    def add_ag_dvars_xr(self, yr: int, ag_dvars: np.ndarray):
        self.ag_dvars_2D_reproj_match[yr] = self.reproj_match_ag_dvar(self.ag_dvars[yr])
        
    def add_am_dvars_xr(self, yr: int, am_dvars: np.ndarray):
        self.am_dvars_2D_reproj_match[yr] = self.reproj_match_am_dvar(self.ag_man_dvars[yr])
        
    def add_non_ag_dvars_xr(self, yr: int, non_ag_dvars: np.ndarray):
        self.non_ag_dvars_2D_reproj_match[yr] = self.reproj_match_non_ag_dvar(self.non_ag_dvars[yr])
        
        
    # Functions to reproject and match the dvars to the target map
    def reproj_match_ag_dvar(self, ag_dvars, reprj_id:str=f'{settings.INPUT_DIR}/bio_id_map.nc', reproj_ref:str=f'{settings.INPUT_DIR}/bio_mask.nc'):

        target_id_map = xr.open_dataset(reprj_id)['data']
        target_ref_map = xr.open_dataset(reproj_ref)['data']
        ag_dvars = self.ag_dvars_to_xr(ag_dvars)
        
        # Parallelize the reprojection and matching
        def reproj_match(dvar, lm, lu):
            dvar = self.dvar_to_2D(dvar)
            dvar = self.dvar_to_full_res(dvar)
            dvar = self.bincount_avg(target_id_map, dvar)
            dvar = dvar.reshape(target_ref_map.shape)
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'lm':[lm], 'lu':[lu]})
        
        tasks = [delayed(reproj_match)(ag_dvars.sel(lm=lm, lu=lu), lm, lu) 
                 for lm,lu in product(self.LANDMANS, self.AGRICULTURAL_LANDUSES)]
        
        return  xr.combine_by_coords(
            [i for i in tqdm(Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks), total=len(tasks))]
            )
    

    def reproj_match_am_dvar(self, am_dvars, reprj_id:str=f'{settings.INPUT_DIR}/bio_id_map.nc', reproj_ref:str=f'{settings.INPUT_DIR}/bio_mask.nc'):
        target_id_map = xr.open_dataset(reprj_id)['data']
        target_ref_map = xr.open_dataset(reproj_ref)['data']
        am_dvars = self.am_dvars_to_xr(am_dvars)
        
        # Parallelize the reprojection and matching
        def reproj_match(dvar, am, lm, lu):
            dvar = self.dvar_to_2D(dvar)
            dvar = self.dvar_to_full_res(dvar)
            dvar = self.bincount_avg(target_id_map, dvar)
            dvar = dvar.reshape(target_ref_map.shape)
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'am':[am], 'lm':[lm], 'lu':[lu]})
        
        # Parallelize the reprojection and matching
        tasks = [delayed(reproj_match)(am_dvars.sel(am=am, lm=lm, lu=lu), am, lm, lu) 
                 for am,lm,lu in product(AG_MANAGEMENTS_TO_LAND_USES, self.LANDMANS, self.AGRICULTURAL_LANDUSES)]
        
        return xr.combine_by_coords(
            [i for i in tqdm(Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks), total=len(tasks))]
            )
    
    
    
    def reproj_match_non_ag_dvar(self, non_ag_dvars, reprj_id:str=f'{settings.INPUT_DIR}/bio_id_map.nc', reproj_ref:str=f'{settings.INPUT_DIR}/bio_mask.nc'):
        target_id_map = xr.open_dataset(reprj_id)['data']
        target_ref_map = xr.open_dataset(reproj_ref)['data']
        non_ag_dvars = self.non_ag_dvars_to_xr(non_ag_dvars)
        
        # Parallelize the reprojection and matching
        def reproj_match(dvar, lu):
            dvar = self.dvar_to_2D(dvar)
            dvar = self.dvar_to_full_res(dvar)
            dvar = self.bincount_avg(target_id_map, dvar)
            dvar = dvar.reshape(target_ref_map.shape)
            dvar = xr.DataArray(dvar, dims=('y', 'x'), coords={'y': target_ref_map['y'], 'x': target_ref_map['x']})
            return dvar.expand_dims({'lu':[lu]})
        
        tasks = [delayed(reproj_match)(non_ag_dvars.sel(lu=lu),  lu) for lu in self.NON_AGRICULTURAL_LANDUSES]
        
        return xr.combine_by_coords(
            [i for i in tqdm(Parallel(n_jobs=10, backend='threading', return_as='generator')(tasks), total=len(tasks))]
            )
        


    
    # Functions to convert dvars to xarray
    def ag_dvars_to_xr(self, ag_dvars: np.ndarray):
        ag_dvar_xr = xr.DataArray(
            ag_dvars, 
            dims=('lm', 'cell', 'lu'),
            coords={
                'lm': self.LANDMANS,
                'cell': np.arange(ag_dvars.shape[1]),
                'lu': self.AGRICULTURAL_LANDUSES
            }   
        ).reindex(lu=self.AGRICULTURAL_LANDUSES) # Reorder the dimensions to match the LUTO variable array indexing
        
        return ag_dvar_xr
    
    
    def am_dvars_to_xr(self, am_dvars: np.ndarray):
        am_dvar_l = []
        for am in am_dvars.keys():  
            am_dvar_xr = xr.DataArray(
                am_dvars[am], 
                dims=('lm', 'cell', 'lu'),
                coords={
                    'lm': self.LANDMANS,
                    'cell': np.arange(am_dvars[am].shape[1]),
                    'lu': self.AGRICULTURAL_LANDUSES})
            
            # Expand the am dimension, the dvar is a 4D array [am, lu, cell, lu]
            am_dvar_xr = am_dvar_xr.expand_dims({'am':[am]})   
            am_dvar_l.append(am_dvar_xr)
                
        return xr.combine_by_coords(am_dvar_l).reindex(
            am=AG_MANAGEMENTS_TO_LAND_USES.keys(), 
            lu=self.AGRICULTURAL_LANDUSES, 
            lm=self.LANDMANS)   # Reorder the dimensions to match the LUTO variable array indexing

        
    def non_ag_dvars_to_xr(self, non_ag_dvars: np.ndarray):
        non_ag_dvar_xr = xr.DataArray(
            non_ag_dvars, 
            dims=('cell', 'lu'),
            coords={
                'cell': np.arange(non_ag_dvars.shape[0]),
                'lu': self.NON_AGRICULTURAL_LANDUSES})
        
        return non_ag_dvar_xr.reindex(
            lu=self.NON_AGRICULTURAL_LANDUSES) # Reorder the dimensions to match the LUTO variable array indexing
            
    
    # Convert dvar to its 2D representation
    def dvar_to_2D(self, map_:np.ndarray)-> np.ndarray:
        map_resfactored = self.LUMAP_2D_RESFACTORED.copy().astype(np.float32)
        np.place(map_resfactored, (map_resfactored != self.MASK_LU_CODE) & (map_resfactored != self.NODATA), map_) 
        return map_resfactored
    
    
    # Upsample dvar to its full resolution representation
    def dvar_to_full_res(self, dvar_2D:np.ndarray) -> np.ndarray:
        dense_2D_shape = self.NLUM_MASK.shape
        dense_2D_map = np.repeat(np.repeat(dvar_2D, settings.RESFACTOR, axis=0), settings.RESFACTOR, axis=1)       

        # Adjust the dense_2D_map size if it differs from the NLUM_MASK
        if dense_2D_map.shape[0] > dense_2D_shape[0] or dense_2D_map.shape[1] > dense_2D_shape[1]:
            dense_2D_map = dense_2D_map[:dense_2D_shape[0], :dense_2D_shape[1]]
            
        if dense_2D_map.shape[0] < dense_2D_shape[0] or dense_2D_map.shape[1] < dense_2D_shape[1]:
            pad_height = dense_2D_shape[0] - dense_2D_map.shape[0]
            pad_width = dense_2D_shape[1] - dense_2D_map.shape[1]
            dense_2D_map = np.pad(
                dense_2D_map, 
                pad_width=((0, pad_height), (0, pad_width)), 
                mode='edge')

        # Apply the masks
        filler_mask = self.LUMAP_2D != self.MASK_LU_CODE
        dense_2D_map = np.where(filler_mask, dense_2D_map, self.MASK_LU_CODE)
        dense_2D_map = np.where(self.NLUM_MASK, dense_2D_map, self.NODATA)
        return dense_2D_map
    
    
    # Upsample dvar to its full resolution representation
    def upsample_dvar(self, map_:np.ndarray, x:np.ndarray, y:np.ndarray)-> xr.DataArray:
        # Get the coords of the map_
        coords = dict(map_.coords)
        del coords['cell']
        
        # Up sample the arr as RESFACTOR=1
        if settings.RESFACTOR > 1:   
            map_ = self.dvar_to_2D(map_)
            map_ = self.dvar_to_full_res(map_)
        else:
            empty_map = np.full_like(self.NLUM_MASK, fill_value=self.NODATA, dtype=np.float32)
            np.place(empty_map, self.NLUM_MASK, self.LUMAP_NO_RESFACTOR) 
            np.place(empty_map, (empty_map != self.MASK_LU_CODE) & (empty_map != self.NODATA), map_)
            map_ = empty_map
        
        # Convert to xarray
        map_ = xr.DataArray(map_, dims=('y','x'), coords={'y': y, 'x': x})
        map_ = map_.expand_dims(coords)
        return map_


    # Calculate the average value of dvars within the target bin       
    def bincount_avg(self, target_id_map, dvar):
        
        # Only dvar > 0 are necessary for the calculation
        valid_mask = dvar > 0

        # Flatten arries
        bin_flatten = target_id_map.values[valid_mask]
        weights_flatten = dvar[valid_mask]
        bin_occ = np.bincount(bin_flatten, minlength=target_id_map.max().values + 1)
        bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=target_id_map.max().values + 1)
    
        # Calculate the average value of each bin, ignoring division by zero (which will be nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_avg = (bin_sum / bin_occ).astype(np.float32)
            bin_avg = np.nan_to_num(bin_avg)
        
        return bin_avg