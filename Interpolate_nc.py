import xarray as xr
import rioxarray as rxr
import numpy as np


# Load the data
f_base = 'N:/Planet-A/LUF-Modelling/LUTO2_JZ/biodiversity_data/data'

bio_nc = xr.open_dataset(f'{f_base}/bio_BCC.CSM2.MR.nc', chunks='auto', engine='h5netcdf')


sel = bio_nc.sel(x=slice(112.9, 113.5), y=slice(-10.02, -10.5))

























