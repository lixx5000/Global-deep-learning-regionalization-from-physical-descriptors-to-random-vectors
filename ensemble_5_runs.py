from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import os
from pathlib import Path, PosixPath
import math as math
import sqlite3
import pandas as pd
from datetime import datetime
import pickle
from joblib import Parallel, delayed
import multiprocessing
import argparse
from typing import Dict, List, Tuple


# get arguments. 
parser = argparse.ArgumentParser()
parser.add_argument('--output_p_direc', type = str)
parser.add_argument('--ensemble_file_name', type = str)
cfg = vars(parser.parse_args())



# go to the output pickle file directly. 
if 'mixed' in cfg["output_p_direc"] or 'num_of_sf' in cfg["output_p_direc"]:
    p_files = list(Path(cfg["output_p_direc"]).glob('*/*/*.p')) # retrieve the pickle file. It has one more level indicating the seed number that generates the Gaussian number. 
else:
    p_files = list(Path(cfg["output_p_direc"]).glob('*/*.p')) # retrieve the pickle file. 
print ('There are ' + str(len(p_files)) + ' different runs to ensemble')
    
basins = list(pd.read_pickle(p_files[0]).keys())  # The keys in the discovered pickles are basin ids, because they have the same number of basins, so we'll use the only the first one to retrieve the basin ids. 

def merge_each_id(basin: str) -> pd.DataFrame:
    """
    ensemble for each basin across 5 runs. 
    """

    p_data = [pd.read_pickle(p) for p in p_files]
    
    ray = pd.concat([p_pickle[basin] for p_pickle in p_data], join = 'outer', axis = 0)
    ray = ray.groupby(ray.index).mean()
    
    print (basin + ': done!')
    
    return ray

# The ensemble process is parallelized to speed up computation. 
num_cores=multiprocessing.cpu_count()
ensemble_all = Parallel(n_jobs=num_cores)(delayed(merge_each_id)(basin) for basin in basins)
    
ensemble_dict = {}
for basin in basins:
    ensemble_dict[basin] = ensemble_all[basins.index(basin)]

with open(Path(cfg['output_p_direc']) / cfg['ensemble_file_name'], 'wb') as handle:
    pickle.dump(ensemble_dict, handle)