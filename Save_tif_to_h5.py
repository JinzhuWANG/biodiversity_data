import pandas as pd
import h5py

from tqdm.auto import tqdm
from joblib import Parallel, delayed
from codes.helper_func import get_all_path, warp_raster


# # Search for all tif files and save to csv
# bio_path = r'N:\Data-Master\Biodiversity\Environmental-suitability\Annual-species-suitability_20-year_snapshots_5km'
# get_all_path(bio_path)


# Load the csv file
df = pd.read_csv('data/all_suitability_tifs.csv')
ref_tif = 'data/lumap_2010_fullres.tiff'



# Function to write data to h5 file
def write_h5(in_path, h5_location, out_path):
    data = warp_raster(in_path, ref_tif)
    with h5py.File(out_path, 'a') as f:
        f.create_dataset(h5_location, data=data, dtype='int8', compression=9)
    return None


tasks = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    ssp = row['ssp']
    model = row['model']
    mode = row['mode']
    group = row['group']
    species = row['species']
    year = row['year']
    h5_location = f'{ssp}/{mode}/{group}/{species}/{year}'
    
    if year == 'historic':
        continue
    tasks.append(delayed(write_h5)(row['path'], h5_location, f"data/bio_suitability_{model}.h5"))



pbar = tqdm(total=len(tasks))
parallel = Parallel(n_jobs=df['model'].nunique()*2, backend='threading', return_as='generator')
parallel_generator = parallel(tasks)
for out in parallel_generator:
    pbar.update(1)
pbar.close()



