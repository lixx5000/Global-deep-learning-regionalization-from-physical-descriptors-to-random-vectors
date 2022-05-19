"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features, calculate_mean_and_std)
from papercode.ealstm import EALSTM, FMLSTM, SRLSTM_EA
from papercode.lstm import LSTM
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list, get_ray_basin, get_eval_basin_list, get_531_basins


###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 256, 
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 20,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 270,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'), 
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),  # '30092008'. 
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30092008', format='%d%m%Y')
}

# check if GPU is available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
print (DEVICE)


###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness"])
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--cluster', type=str, help="cluster #: . Basins within the cluster as specified in the folder. ")
    parser.add_argument('--attri_rand_seed', type=int, required=False, help="Random seed to randomlize attribute seeds")
    parser.add_argument('--rand_feat_num', type=int, required=False, help="# of random features")
    parser.add_argument('--cache_data',
                        type=bool,
                        default=False,
                        help="If True, loads all data into memory")
    parser.add_argument('--num_workers',
                        type=int,
                        default=12,
                        help="Number of parallel threads for data loading")
    parser.add_argument("--one_hot", 
                       type = bool, default = False, help = "If True, we'll use the one-hot vector to characterize catchments.")    
    parser.add_argument('--with_embeddings', 
                       type = bool, default = False, help = "If True, an additional embedding layer is created between static features and LSTM cell")
    parser.add_argument("--FM_LSTM", 
                       type = bool, default = False, help = "If True, the FM(Feature Modulation)-LSTM will be specified. ")
    parser.add_argument('--no_static',
                        type=bool,
                        default=False,
                        help="If True, trains LSTM without static features")
    parser.add_argument('--mixed', 
                       type = bool, 
                       default = False, help = 'If true, the static vectors are mixed Gaussian vectors (both Gaussian vectors and physical descriptors are used. )')
    parser.add_argument('--concat_static',
                        type=bool,
                        default=False,
                        help="If True, train LSTM with static feats concatenated at each time step")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=False,
                        help="If True, uses mean squared error as loss function.")
    cfg = vars(parser.parse_args())
    
    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))
        cfg['attri_rand_seed'] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])

    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
        
    # dump the SCALER pickle file. 
    SCALER = calculate_mean_and_std(camels_root = cfg['camels_root'], 
                                    train_dates = [cfg["train_start"], cfg["train_end"]], 
                                   basins = get_basin_list(cfg['cluster']))
    file_name = Path(__file__).absolute().parent / ("data/SCALER.p")
    with (file_name).open('wb') as fp:
        pickle.dump(SCALER, fp)      
    
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    
    lstm_prefix = 'ea'
    if cfg['concat_static']:
        lstm_prefix = 'ct'
    if cfg['with_embeddings']:
        lstm_prefix = 'sr'
    if cfg['FM_LSTM']:
        lstm_prefix = 'fm'
    
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / str(cfg['cluster']) / lstm_prefix / 'physics' /('seed'+str(cfg["seed"]))
    
    if cfg['no_static']:
        cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / str(cfg['cluster']) / 'no_static' /  ('seed'+str(cfg["seed"]))
    
    if cfg['mixed']:
        cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / str(cfg['cluster']) / lstm_prefix / 'mixed' / str(cfg['rand_feat_num']) / str(cfg['attri_rand_seed']) / ('seed'+str(cfg["seed"]))
    elif cfg['one_hot']:
        cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / str(cfg['cluster']) / lstm_prefix / 'one_hot' / ('seed'+str(cfg["seed"]))
    elif cfg['rand_feat_num'] is not None:
        cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / str(cfg['cluster']) / lstm_prefix / 'num_of_sf' / str(cfg['rand_feat_num']) / str(cfg['attri_rand_seed']) / ('seed'+str(cfg["seed"]))
    
    
    # Now we'll specify the random vector directory. 
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"], basins = basins,
                          rand_feat_num = cfg['rand_feat_num'], attri_rand_seed = cfg['attri_rand_seed'], 
                          one_hot = cfg['one_hot'], mixed = cfg['mixed'])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(camels_root=cfg["camels_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])

    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connceted layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False, 
                 add_embedding: bool = False, 
                 fm: bool = False):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        add_embedding: bool
            If True, runs the LSTM with an embedding layer between xs and LSTM input gates. 
        fm: bool
            If True, runs the FM-LSTM (Feature Modulation LSTM). 
        """
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static
        self.add_embedding = add_embedding
        self.fm = fm
        
        # Define the LSTM structure. 
        if self.concat_static or self.no_static:
            self.lstm = LSTM(input_size=input_size_dyn,
                             hidden_size=hidden_size,
                             initial_forget_bias=initial_forget_bias)
        elif self.add_embedding:
            self.lstm = SRLSTM_EA(input_size_dyn=input_size_dyn,
                                  input_size_stat=input_size_stat,
                                  hidden_size=hidden_size,
                                  initial_forget_bias=initial_forget_bias, 
                                  ann_1=512)  # we mannually put the dim of the inserted embedding layer to be 512 becuase of the EG-512. See manuscript. 
        elif self.fm:
            self.lstm = FMLSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)
        else:
            self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)
        
        

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch,Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        return out, h_n, c_n


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = get_basin_list(cfg["cluster"])

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare PyTorch DataLoader
    ds = CamelsH5(h5_file=cfg["train_file"],
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=cfg["concat_static"],
                  cache=cfg["cache_data"],
                  no_static=cfg["no_static"])
    
    loader = DataLoader(ds,
                        batch_size=cfg["batch_size"],
                        shuffle=True,
                        num_workers=cfg["num_workers"])

    # create model and optimizer
    xs = load_attributes(db_path = cfg['db_path'], basins = basins)  # this loaded xs will determine the input_size_static
    input_size_stat = 0 if cfg['no_static'] else xs.shape[1]  # the number of columns in the df is the number of static features. 
    
    input_size_dyn = 5 if (cfg["no_static"] or not cfg["concat_static"]) else (input_size_stat + 5)  
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=cfg["hidden_size"],
                  initial_forget_bias=cfg["initial_forget_gate_bias"],
                  dropout=cfg["dropout"],
                  concat_static=cfg["concat_static"],
                  no_static=cfg["no_static"], 
                  add_embedding=cfg['with_embeddings'], 
                  fm=cfg['FM_LSTM']
                 ) 
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))     # multiple GPU to speed up if accessible. 
    model = model.to(DEVICE)   # multiple GPU
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    # define loss function
    if cfg["use_mse"]:
        loss_func = nn.MSELoss()
    else:
        loss_func = NSELoss()
        
    # reduce learning rates after each 10 epochs
    learning_rates = {11: 5e-4}
    
    tl_ls = []
    for epoch in range(1, cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        tl = train_epoch(model, optimizer, loss_func, loader, cfg, epoch, cfg["use_mse"])
        tl_ls.append(tl)    

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))
    
    loss_df = pd.DataFrame({'total_loss': tl_ls})
    
    loss_df.to_csv(str(cfg["run_dir"]) + '/loss.csv')
    
    # evaluation. ------------
    db_path = str(cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins) 
    
    means = attributes.mean()
    stds = attributes.std()
    
    if np.sum(attributes.sum(axis = 1) == 1) == len(basins):
        print ('it is one-hot, no need to normalize static features')
        means = pd.DataFrame(np.zeros(attributes.mean().shape))[0]
        stds = pd.DataFrame(np.ones(attributes.std().shape))[0]
        
        means.index = [str(i) for i in range(len(basins))]
        stds.index = [str(i) for i in range(len(basins))]
    
    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=cfg["concat_static"],
                            db_path=db_path)
        
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)

        preds, obs = evaluate_basin(model, loader)

        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

        results[basin] = df

# store these results per basin.
#     print (user_cfg['run_dir'])
    file_name = cfg["run_dir"] / ("output.p")
    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)




def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool):
    """Train model for a single epoch.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    optimizer : torch.optim.Optimizer
        Optimizer used for weight updating
    loss_func : nn.Module
        The loss function, implemented as a PyTorch Module
    loader : DataLoader
        PyTorch DataLoader containing the training data in batches.
    cfg : Dict
        Dictionary containing the run config
    epoch : int
        Current Number of epoch
    use_mse : bool
        If True, loss_func is nn.MSELoss(), else NSELoss() which expects addtional std of discharge
        vector

    """
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    total_loss = .0
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        if len(data) == 3:
            x, y, q_stds = data
            x, y, q_stds = x.to(DEVICE), y.to(DEVICE), q_stds.to(DEVICE)     # to(DEVICE)
            predictions = model(x)[0]

        # forward pass through EALSTM
        elif len(data) == 4:
            x_d, x_s, y, q_stds = data
            x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)   # to(DEVICE)
            predictions = model(x_d, x_s[:, 0, :])[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(DEVICE)           #to(DEVICE)
            loss = loss_func(predictions, y, q_stds)
                
            total_loss += loss.item()

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

#         pbar.set_postfix_str(f"Loss: {loss.item():5f}" + f" NSE : {temp_nse_loss.item():5f}" + f" REC: {temp_rec_loss.item():5f}")
        pbar.set_postfix_str(f"Loss: {loss.item():5f}")
        
    return total_loss/len(pbar)


def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list(user_cfg["cluster"])

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins)    
    
    means = attributes.mean()
    stds = attributes.std()
    
    
    # create model and optimizer
    if np.sum(attributes.sum(axis = 1) == 1) == len(basins):
        print ('it is one-hot, no need to normalize static features')
        self.df = df
        means = pd.DataFrame(np.zeros(attributes.mean().shape))[0]
        stds = pd.DataFrame(np.ones(attributes.std().shape))[0]
        
        means.index = [str(i) for i in range(len(basins))]
        stds.index = [str(i) for i in range(len(basins))]
    
    
    # create model and optimizer
    input_size_stat = 0 if run_cfg['no_static'] else attributes.shape[1]  # the number of columns in the df is the number of static features. 
    
    input_size_dyn = 5 if (run_cfg["no_static"] or not run_cfg["concat_static"]) else (input_size_stat + 5)  
    model = Model(input_size_dyn=input_size_dyn,
                  input_size_stat=input_size_stat,
                  hidden_size=run_cfg["hidden_size"],
                  initial_forget_bias=run_cfg["initial_forget_gate_bias"],
                  dropout=run_cfg["dropout"],
                  concat_static=run_cfg["concat_static"],
                  no_static=run_cfg["no_static"], 
                  add_embedding=run_cfg['with_embeddings'], 
                  fm=run_cfg['FM_LSTM'])
    
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))     # multiple GPU if accessoble
    model = model.to(DEVICE)   
                                                               
    # load trained model
    weight_file = user_cfg["run_dir"] / 'model_epoch20.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=run_cfg["concat_static"],
                            db_path=db_path)

        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=4)

        preds, obs = evaluate_basin(model, loader)

        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

        results[basin] = df

# store these results per basin.
    print (user_cfg['run_dir'])
    file_name = user_cfg["run_dir"] / ("output.p")
    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 2:
                x, y = data
                x, y = x.to(DEVICE), y.to(DEVICE)   
                p = model(x)[0]
            elif len(data) == 3:
                x_d, x_s, y = data
                x_d, x_s, y = x_d.to(DEVICE), x_s.to(DEVICE), y.to(DEVICE)
                p = model(x_d, x_s[:, 0, :])[0]                
                 

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    return preds, obs

def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"lstm_no_static_seed{run_cfg['seed']}.p"
    else:
        if run_cfg["concat_static"]:
            file_name = user_cfg["run_dir"] / f"lstm_seed{run_cfg['seed']}.p"
        else:
            file_name = user_cfg["run_dir"] / f"ealstm_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
