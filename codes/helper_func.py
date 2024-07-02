import os
import re
import numpy as np
import pandas as pd

from scipy.ndimage import distance_transform_edt
from tqdm.auto import tqdm


def replace_mask_with_nearest(map_: np.ndarray, mask) -> np.ndarray:
    """
    Replaces invalid values in the input array with the nearest non-filler values.

    Parameters:
        map_ (np.ndarray, 2D): The input array.
        mask (np.ndarray, bool): The boolean mask indicating the invalid values in the input array.

    Returns:
        np.ndarray (2D): The array with invalid values replaced by the nearest non-invalid values.
    """

    # Perform distance transform on the mask
    _, nearest_indices = distance_transform_edt(mask, return_indices=True)
    # Replace the invalid values with the nearest non-invalid values
    map_[mask] = map_[tuple(nearest_indices[:, mask])]
    
    return map_




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
    
    
    reg_year = re.compile(r'_(\d{4})_').findall(row['path'])[0]
    
    if int(reg_year) < 2010:
        return ['historic', 'historic', reg_year, 'historic']
    
    reg_model = re.compile(rf'{row["species"]}_(.*)_ssp').findall(row['path'])[0]
    reg_ssp = re.compile(r'_(ssp\d*)_').findall(row['path'])[0]
    reg_mode = re.compile(r'km_(.*).tif').findall(row['path'])[0]
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


