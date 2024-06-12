import xarray as xr
import numpy as np

from tqdm.auto import tqdm
from joblib import Parallel, delayed
from dask.diagnostics import ProgressBar


# Read data
interplolated = xr.open_dataset('data/bio_ensemble_interpolated.nc', engine='h5netcdf', chunks='auto')['data']
interplolated['x'] = interplolated['x'].compute()


coord_sparse = np.load("data/coord_lon_lat_res10.npy")
coord_dens = np.load("data/coord_lon_lat_res1.npy")

mask = xr.DataArray(
    np.isin(interplolated.x, coord_sparse) & np.isin(interplolated.y, coord_sparse),
    coords={'points': interplolated['points']})

filtered_data = interplolated.where(mask, drop=True)









