Create LSTM family models using Gaussian vectors. 

The Gaussian vectors are random numbers drawn from a normal Gaussian distribution, that is, zero mean and unit variance. For each catchment, it will be assigned a d-dimension vectors. d is detemrined empirically as the one yielding the best performance. 
For EA-LSTM and FM-LSTM, d is 256 or 512. For CT-LSTM, d is 16 or 32. 

#### Content of the Repository

#### Run the local Code
Train and test model. For ease of both training and testing, we put these 2 procedures together in one run. The periods of training and testing are specified in main.py. You'll need to run the following line of code from the terminal. 
```
python python main.py train --rand_feat_num NUMBER  --cluster NUMBER --camels_root /path/to/camels --cache_data True --num_workers 12--attri_rand_seed NUMBER
```
A single LSTM (EA or CT) will be trained with a specified d Gaussian dimension on a specified set of basins (either subset of 531 basins or the total 531 basins).
- ```--rand_feat_num``` the dimension of d-dimension Gaussian vector. 
- ```--attri_rand_seed``` A seed number for generating the Gaussian vector. 
- ```--cluster file_name``` The cluster number of catchments we'll use to trian the global model. ```file_name``` argument specifies the name of the basin_list. For example, if you want to use 531 basins as stored in 531.txt under data folder. You'll need to type ```--cluster 531```. Otherwise, if you want to train the model on any subset of basins, please save those basin ids as a txt file under ```data``` folder and then specify the txt file name as the argument. 
- ```--camels-root```, ```--cache_data True```, ```--num_workers NUMBER```, ```--concat_static True```, ```--use_mse True``` The details documenting those commands can be found in https://github.com/kratzert/ealstm_regional_modeling. For readers' conveneince, we also attached them at the end of this session. 



