import numpy as np

from codes import ag_dvar_to_bio_map, ag_to_xr, am_dvar_to_bio_map, am_to_xr, non_ag_dvar_to_bio_map, non_ag_to_xr
from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

# Read in fake data
max_workers = 10
res_factor = 5
year = 2050
data = Data(res_factor)


# Get dvars
ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{year}.npy')         
am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{year}.npy')  
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}
non_ag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{year}.npy')


# dvar to xarray 
ag_dvar = ag_to_xr(data, ag_dvar)
am_dvar = am_to_xr(data, am_dvar)
non_ag_dvar = non_ag_to_xr(data, non_ag_dvar)



# Reproject and match dvars to the bio map
ag_dvar_2D_5km = ag_dvar_to_bio_map(data, ag_dvar, res_factor, max_workers)
am_dvar_2D_5km = am_dvar_to_bio_map(data, am_dvar, res_factor, max_workers)
non_ag_dvar_2D_5km = non_ag_dvar_to_bio_map(data, non_ag_dvar, res_factor, max_workers)





