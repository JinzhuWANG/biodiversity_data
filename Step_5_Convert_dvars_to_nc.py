import xarray as xr
import numpy as np

from itertools import product
from codes import ag_to_xr, am_to_xr, match_lumap_biomap, non_ag_to_xr
from codes.fake_func import Data
from luto.ag_managements import AG_MANAGEMENTS_TO_LAND_USES

res_factor = 10
year = 2050
data = Data(res_factor)

ag_dvar = np.load(f'data/dvars/res{res_factor}/ag_X_mrj_{year}.npy')          # mrj
ag_dvar = ag_to_xr(data, ag_dvar)
ag_dvar = ag_dvar.reindex(lu=data.AGRICULTURAL_LANDUSES)


am_dvar = {k: np.load(f'data/dvars/res{res_factor}/ag_man_X_mrj_{k.lower().replace(' ', '_')}_{year}.npy') 
           for k in AG_MANAGEMENTS_TO_LAND_USES.keys()}
am_dvar = xr.combine_by_coords([am_to_xr(data, k, v) for k,v in am_dvar.items()])
am_dvar = am_dvar.reindex(am=AG_MANAGEMENTS_TO_LAND_USES.keys())


nonag_dvar = np.load(f'data/dvars/res{res_factor}/non_ag_X_rk_{year}.npy')
nonag_dvar = non_ag_to_xr(data, nonag_dvar)
nonag_dvar = nonag_dvar.reindex(lu=data.NON_AGRICULTURAL_LANDUSES)





map_2d = []
for i,j in product(ag_dvar['lm'].values, ag_dvar['lu'].values):
    map_ = ag_dvar.sel(lm=i, lu=j)
    map_ = match_lumap_biomap(data, map_, res_factor)
    map_ = map_.expand_dims({'lm': [i], 'lu': [j]})
    map_2d.append(map_)
map_2d = xr.combine_by_coords(map_2d)