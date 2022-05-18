"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import sqlite3
from pathlib import Path, PosixPath
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from numba import njit
import random
from tqdm import tqdm

# CAMELS catchment characteristics ignored in this study
INVALID_ATTR = [
    'gauge_name', 'area_geospa_fabric', 'geol_1st_class', 'glim_1st_class_frac', 'geol_2nd_class',
    'glim_2nd_class_frac', 'dom_land_cover_frac', 'dom_land_cover', 'high_prec_timing',
    'low_prec_timing', 'huc', 'q_mean', 'runoff_ratio', 'stream_elas', 'slope_fdc',
    'baseflow_index', 'hfd_mean', 'q5', 'q95', 'high_q_freq', 'high_q_dur', 'low_q_freq',
    'low_q_dur', 'zero_q_freq', 'geol_porostiy', 'root_depth_50', 'root_depth_99', 'organic_frac',
    'water_frac', 'other_frac'
]



def get_531_basins() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path('/home/nieberj/lixx5000/Camels/data/basin_list.txt')
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins
basins = get_531_basins()


def add_camels_attributes(camels_root: PosixPath, db_path: str = None, basins: List = None,
                          rand_feat_num: int = None, attri_rand_seed: int = None, one_hot: bool = False, mixed: bool = False):
    """Load catchment characteristics from txt files and store them in a sqlite3 table
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    db_path : str, optional
        Path to where the database file should be saved. If None, stores the database in the 
        `data` directory in the main folder of this repository., by default None
    basins: List, 
        The list of the basins to load. 
    rand_feat_num: int
        the number of generated Gaussian number if specified to do so. 
    attri_rand_seed: int
        The seed number of the corresponding Gaussian number generator. 
    one_hot: bool
        If True, a one-hot vector denoting each basin is generated. 
    mixed: bool
        If True, it will create a mixed Gaussian vector (27 phyiscal descriptors + Gaussian vector). The total dimension will be specified by the rand_feat_num in this case. 
    
    Raises
    ------
    RuntimeError
        If CAMELS attributes folder could not be found.
    """
       
    attributes_path = Path(camels_root) / 'camels_attributes_v2.0'

    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('camels_*.txt')
    
    df = None
    for f in txt_files:
        df_temp = pd.read_csv(f, sep=';', header=0, dtype={'gauge_id': str})
        df_temp = df_temp.set_index('gauge_id')

        if df is None:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp], axis=1)

    # convert huc column to double digit strings
    df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
    df = df.drop('huc_02', axis=1)
    
    # subset df by the basin List. 
    df = df.loc[basins]
    n_dim = len(basins)
    
    # df will be replaced by Gaussian vectors, one-hot vectors, or mixed-Gaussian vectors. 
    if rand_feat_num is not None: 
        # Assign Gaussian distributed random numbers to df dimenison by dimension. 
        # Now you want to randomlize them given only run_cfg['rand_feat_num'] features. 
        np.random.seed(attri_rand_seed)
        if mixed:
            # random numbers are only needed for the (rand_feat_num - 27) dimensions.
            random_df = pd.DataFrame(np.random.normal(0, 1, size = (n_dim, rand_feat_num - 27)))
            random_df.index = df.index
            df = df.merge(random_df, left_index = True, right_index = True)
        else: 
            # all dimensions will be Guassian distributed random vectors. 
            random_df = pd.DataFrame(np.random.normal(0, 1, size = (n_dim, rand_feat_num)))
            random_df.index = df.index
            df = random_df.copy()   
    elif one_hot:
        df = np.zeros((n_dim, n_dim))
        for i in range(n_dim):
            df[i, i%n_dim] = 1
        df = pd.DataFrame(df)

    df['gauge_id'] = basins
    df = df.set_index('gauge_id')

    if db_path is None:
        db_path = str(Path(__file__).absolute().parent.parent / 'data' / 'attributes.db')

    with sqlite3.connect(db_path) as conn:
        # insert into databse
        df.to_sql('basin_attributes', conn)

    print(f"Sucessfully stored basin attributes in {db_path}.")


def load_attributes(db_path: str,
                    basins: List,
                    keep_features: List = None) -> pd.DataFrame:
    """Load attributes from database file into DataFrame

    Parameters
    ----------
    db_path : str
        Path to sqlite3 database file
    basins : List
        List containing the 8-digit USGS gauge id
    keep_features : List
        If a list is passed, a pd.DataFrame containing these features will be returned. By default,
        returns a pd.DataFrame containing the features used for training.

    Returns
    -------
    pd.DataFrame
        Attributes in a pandas DataFrame. Index is USGS gauge id. Latitude and Longitude are
        transformed to x, y, z on a unit sphere.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM 'basin_attributes'", conn, index_col='gauge_id')

    # drop rows of basins not contained in data set
    drop_basins = [b for b in df.index if b not in basins]
    df = df.drop(drop_basins, axis=0)

    # drop lat/lon col only if lat/lon are in the columns
    if 'gauge_lat' in df.columns:
        df = df.drop(['gauge_lat', 'gauge_lon'], axis=1)

    # drop invalid attributes
    if keep_features is not None:
        drop_names = [c for c in df.columns if c not in keep_features]
    else:
        drop_names = [c for c in df.columns if c in INVALID_ATTR]

    df = df.drop(drop_names, axis=1)

    return df


def normalize_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Normalize features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to normalize
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Normalized features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    
    SCALER = pd.read_pickle(Path(__file__).absolute().parent.parent / ("data/SCALER.p"))

    if variable == 'inputs':
        feature = (feature - SCALER["input_means"]) / SCALER["input_stds"]
    elif variable == 'output':
        feature = (feature - SCALER["output_mean"]) / SCALER["output_std"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


def rescale_features(feature: np.ndarray, variable: str) -> np.ndarray:
    """Rescale features using global pre-computed statistics.

    Parameters
    ----------
    feature : np.ndarray
        Data to rescale
    variable : str
        One of ['inputs', 'output'], where `inputs` mean, that the `feature` input are the model
        inputs (meteorological forcing data) and `output` that the `feature` input are discharge
        values.

    Returns
    -------
    np.ndarray
        Rescaled features

    Raises
    ------
    RuntimeError
        If `variable` is neither 'inputs' nor 'output'
    """
    
    SCALER = pd.read_pickle(Path(__file__).absolute().parent.parent / ("data/SCALER.p"))
    
    if variable == 'inputs':
        feature = feature * SCALER["input_stds"] + SCALER["input_means"]
    elif variable == 'output':
        feature = feature * SCALER["output_std"] + SCALER["output_mean"]
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return feature


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape data into LSTM many-to-one input samples

    Parameters
    ----------
    x : np.ndarray
        Input features of shape [num_samples, num_features]
    y : np.ndarray
        Output feature of shape [num_samples, 1]
    seq_length : int
        Length of the requested input sequences.

    Returns
    -------
    x_new: np.ndarray
        Reshaped input features of shape [num_samples*, seq_length, num_features], where 
        num_samples* is equal to num_samples - seq_length + 1, due to the need of a warm start at
        the beginning
    y_new: np.ndarray
        The target value for each sample in x_new
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]
    

    return x_new, y_new


def load_forcing(camels_root: PosixPath, basin: str) -> Tuple[pd.DataFrame, int]:
    """Load Maurer forcing data from text files.

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing the Maurer forcing
    area: int
        Catchment area (read-out from the header of the forcing file)

    Raises
    ------
    RuntimeError
        If not forcing file was found.
    """
    forcing_path = camels_root / 'basin_mean_forcing' / 'maurer_extended'  # change to daymet to adopt more available weather drivers. 
    files = list(forcing_path.glob('**/*_forcing_leap.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]   

    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    

    # load area from header
    with open(file_path, 'r') as fp:
        content = fp.readlines()
        area = int(content[2])

    return df, area


def load_discharge(camels_root: PosixPath, basin: str, area: int) -> pd.Series:
    """[summary]

    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    basin : str
        8-digit USGS gauge id
    area : int
        Catchment area, used to normalize the discharge to mm/day

    Returns
    -------
    pd.Series
        A Series containing the discharge values.

    Raises
    ------
    RuntimeError
        If no discharge file was found.
    """
    
    discharge_path = camels_root / 'usgs_streamflow'
    files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
    file_path = [f for f in files if f.name[:8] == basin]
    if len(file_path) == 0:
        raise RuntimeError(f'No file for Basin {basin} at {file_path}')
    else:
        file_path = file_path[0]

    col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
    df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)
    dates = (df.Year.map(str) + "/" + df.Mnth.map(str) + "/" + df.Day.map(str))
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")

    # normalize discharge from cubic feed per second to mm per day
    df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)
    
    return df.QObs


def calculate_mean_and_std(camels_root: PosixPath, train_dates: List, basins: List) -> Dict:
    """
    Mean and std for weather and discharge data. 
    Those calculations will be used for normalizing input and output. 
    
    Parmaeters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    train_dates : List
        [the start date of training period, the ending date of training period]        
        
    Returns
    -------
    A dictionary with mean and stds of input and output. 
    """
    
    df_all_bsns = []
    q_all_bsns = []
    
    for basin_id in tqdm(basins):
        df, area = load_forcing(camels_root = camels_root, basin = basin_id)
        q = load_discharge(camels_root = camels_root, basin = basin_id, area = area)
        
        # subset by dates. 
        df = df.loc[pd.date_range(start = train_dates[0], end = train_dates[1])]
        q = q.loc[pd.date_range(start = train_dates[0], end = train_dates[1])]
        
        df_all_bsns.append(df[['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']])
        q_all_bsns.append(q)
    
    df_all_bsns = pd.concat(df_all_bsns)
    q_all_bsns = pd.concat(q_all_bsns)
    
    # extract their mean and stds. 
    scalers = {
        'input_means': df_all_bsns.mean().values,
        'input_stds': df_all_bsns.std().values,
        'output_mean': np.array(q_all_bsns.mean()),
        'output_std': np.array(q_all_bsns.std())
    }
    
    return scalers
