# Global-deep-learning-regionalization-from-physical-descriptors-to-random-vectors
The code for performing global deep learning regionalization from physical descriptors to random vectors. The paper is in reviewing processes. 

The code for reproducing the workflow is being organized and will be published soon. The scripts are largely based on Kratzert's work on Catchment-Aware LSTMs for Regional Rainfall-Runoff Modeling (https://github.com/kratzert/ealstm_regional_modeling). Minor revisions regarding how to assign random vectors to catchments, which are proposed solutions in the paper, are made and thus are shown in this repo. 

This paper highlights the benefit of using random vector as a substitute strategy for hydrologic global regionalization when catchment static catchment characteristics are uncertain, incompelte, and unavailable. 

About the model environment set-up and data donwloading, please check the https://github.com/kratzert/ealstm_regional_modeling and use the Updated Maurer forcing data to replicate the results.

# Content of the Repository
- data/
  - text files that store the basin IDs. The name of those text files will be arugments that pass into ```--cluster``` arguments. 
- papercode/
  - consains the entire code what main.py will load from. 
- main.py
  - The Main python file used for training, testing the global deep learning models. To run it, you'll need to type a line of commands on the terminal to start training. Note for running convenience, we merged model evaluation within model training, that is, after taining the model, the model prediction (in testing / validation periods) will be in an output pickle file (`output.p`). A loss.csv file records the epoch-wise loss function value.  
- ensemble_5_runs.py
  - Ensemble the model prediction results from multiple runs (averaging them) to yield final model predictions. We ensembled five different runs in this paper. 
- notebook
  - the jupyter notebook that creates the figures in the manuscript and their corresponding data files. (coming shortly)
# Run main.py
### Training
Running main.py will train a global model on only basins contained in a txt file under `data/`. The loss function is the basin average NSE loss. The results will be stored under `runs/`. To specifiy the model set up (architecture and static vector options) as well as other basic arguments, see below.  

Running the experiments listed below will reproduce the results in the paper (the corresponding experiment and model set up has been mentioned in the bolded font). 
|Model architecture       | 27-d Physics descriptors<br />(Default) | Gaussian vectors<br /> (d-dimensional)<br />`--rand_feat_num d` | one hot vector <br /> `--one_hot True`        | mixed Gaussian vectors <br />(d-dimensional)<br />`--mixed True`<br />`--rand_feat_num d`  |no static vector <br /> `--no_static True`  |
|    :---:                |     :---:   |           :---:                                                 |       :---:                                      |                  :---:                                                                     |       :---:                                |
|    EA-LSTM (Default)              |             **EP**                 | **EG-d**|**EO**|**EM-d**||
|    CT-LSTM <br />`--concat_static True`     |    **CP**    |            **CG-d**        |       **CO**          |                  NA                     |        NA             |
|    SR-LSTMEA <br /> `--with_embedding True` |   **PEA**  |             **REA** (d=27) |          NA           |                  NA                     |     NA                |
|    FM-LSTM <br /> `--FM_LSTM True`        |      **FP**        |           **FG-d**               |        **FO**         |                  NA                     |      NA                |


EA-LSTM are the default model architecture while the 27-d physical descriptors are default static vector (**x<sup>s</sup>**) options. The NA only means the corresponding model set up is not included / discussed in the manuscript. For any corresponding model architecture and static vector combinations, their model performance (ensemble version of five different runs) is reported in the paper. To run them in your own, please combine the arguments contained in their corresponding row and column headers, for instance: 
- to run the **CG-d** (EA-LSTM using d-dimensional Guassian vector), run the following line of code from the terminal 
  - `python main.py train --concat_static_static True --rand_feat_num d`  
- **REA** (SR-LSTMEA with 27-d Gaussian vectors) `python main.py train --with_embedding True --rand_feat_num 27` 
- **EO** (EA-LSTM using one-hot vectors) `python main.py train --one_hot True` 
- **EM-d** (EA-LSTM using d-dimensional mixed Gaussian vector) `python main.py train --mixed True --rand_feat_num d` 
 
 In addition, users must provide the two arguments below to train the model:
 - `-- cluster STRING` the name of the txt file under data/. For instance, to train a global model using 531 basins, whose ids are listed in 531.txt, we need `--cluster 531`
 - `--camels_root /path/to/camels` Specify the CAMELS weather data directory. 
 
 To give a complete example on training a global **EG-512** (the EA-LSTM using 512-d Gaussian vectors) on 531 basins in CAMELS, run the following line of code: 
 - `python main.py train --rand_feat_num 512 --camels_root /path/to/camels --cluster 531`
 
 Other arguments are optional and include those having been explained by [Kratzert repository]([url](https://github.com/kratzert/ealstm_regional_model)). For readers convenience, we explained all of them below. 
 - `--seed NUMBER` Train a model using a fixed random seed. 
 - `--cache_data True` Load the entire training data into memory. It needs approximately 50GB of RAM. 
 - `--num_workers NUMBER` The number of parallel threads that load and process inputs. By default it is 12. 
 - `--attri_rand_seed NUMBER` The fixed random seed of the Gaussian generated static vectors (only applied when `--rand_feat_num` is provided)
 - `--use_mse True` If passed, the loss function will be the mean squared error, instead of the basin average NSE loss. For the accompanied paper, this argument is never activated. 

### Evaluation 
The trained model can also be evaluated to give prediction in testing / validation periods. To evaluate the model, run the command line below: 
- `python main.py evaluate --camels_root /path/to/camels --run_dir path/to/model/run --cluster STRING`
Note that this procedure has also been included in the training procedure. In case users might want to do it separately, its illustration is provided. 

### Ensemble the prediction
The results from multiple-runs need to be ensembled (i.e., the average prediction for the same basin has to be given as averages from different runs). To do this, please run the command line below: 
- `python ensemble_5_runs.py --output_p_direc /path/to/the/model/runs/specified/by/model/set/up/ --ensemble_file_name ensembled_file_name`
  - `--output_p_direc` shall not be the final level where the 'output.p' is located, intead, it has to be specifeid only to either the 'static_vector' level or the 'rand_feat_num' level (if this level has been created, see the Results structure section below). 
  - `--ensemble_file_name` is a user-provided string of the final ensembled modle output file name. 

# Results struture
Files under the run/ are organized in this following structure: 
```bash

└── runs
  └── cluster_STRING      # the argument for `--cluster`, which is the basin list txt file name in the data/ folder
      └── model_structure  # options in the 'Model architecture' column of the above table, can only be 'ea' (EA-LSTM), 'ct' (CT-LSTM), 'sr' (SR-LSTMEA), 'fm' (FM-LSTM)  
          └── static vector # options in the static vectors (the column header in the above table), can only be 'physics' (physical descriptors), 'one_hot', 'no_static', 'num_of_sf' (Gaussian vector), 'mixed' (mixed Gaussian vector). 
              └── rand_feat_num               # (deprecated unless `--rand_feat_num` is specified) the dimension of random vectors (i.e., the number specified in the `--rand_feat_num` argument)
                  └── attri_rand_seed         # (deprecated unless `--rand_feat_num` is specified) the seed number for the Gaussian vector generation. 
                      └── seed_NUMBER         # model initialization seed number. 
                          ├── output.p        # model prediction pickle file
                          ├── attributes.db   # static vector 
                          ├── cfg.json        # the configuration file for the model training
                          ├── model_epochx.pt # model paramter files saved after each epoch (x denotes the epoch number)
                          └──  loss.csv       # epoch-wise loss values 


