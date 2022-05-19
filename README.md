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
  - Main python file used for training, testing the global deep learning models. To run it, you'll need to type a line of commands on the terminal to start training. Note for running convenience, we merged model evaluation within model training, that is, after taining the model, the model prediction (in both trianing and testing periods) will be in the output. 
# Run main.py
Running the experiments listed below will reproduce the results in the paper (the corresponding experiment and model set up has been mentioned in the bolded font). 
|Model architecture       | 27-d Physics descriptors<br />(Default) | Gaussian vectors<br /> (d-dimensional)<br />`--rand_feat_num d` | one hot vector <br /> `--one_hot True`        | mixed Gaussian vectors <br />(d-dimensional)<br />`--mixed True`<br />`--rand_feat_num d`  |no static vector <br /> `--no_static True`  |
|    :---:                |     :---:   |           :---:                                                 |       :---:                                      |                  :---:                                                                     |       :---:                                |
|    EA-LSTM (Default)              |             **EP**                 | **EG-d**|**EO**|**EM-d**||
|    CT-LSTM <br />`--concat_static True`     |    **CP**    |            **CG-d**        |       **CO**          |                  NA                     |        NA             |
|    SR-LSTMEA <br /> `--with_embedding True` |   **PEA**  |             **REA** (d=27) |          NA           |                  NA                     |     NA                |
|    FM-LSTM <br /> `--FM_LSTM True`        |      **FP**        |           **FG-d**               |        **FO**         |                  NA                     |      NA                |


EA-LSTM are the default model architecture while the 27-d physical descriptors are default static vector (**x<sup>s</sup>**) options. The NA only means the corresponding model set up is not included / discussed in the manuscript. For any corresponding model architecture and static vector combinations, their model performance (ensemble version of five different runs) is reported in the paper. To run them in your own, please combine the arguments contained in their corresponding row and column headers, for instance: 
- to run the **CG-d** (EA-LSTM using d-dimensional Guassian vector), run the following line of code from the terminal 
  - `python main.py train --camels_root /path/to/CAMELS --concat_static_static True --rand_feat_num d`  
- **REA** (SR-LSTMEA with 27-d Gaussian vectors): `python main.py train --camels_root /path/to/CAMELS --with_embedding True --rand_feat_num 27` 
- **EO** (EA-LSTM using one-hot vectors): `python main.py train --camels_root /path/to/CAMELS --one_hot True` 
- **EM** (EA-LSTM using d-dimensional mixed Gaussian vector), `python main.py train --camels_root /path/to/CAMELS --mixed True --rand_feat_num d` 
# Results struture
