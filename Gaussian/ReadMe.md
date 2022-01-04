Create LSTM family models using Gaussian vectors. 

The Gaussian vectors are random numbers drawn from a normal Gaussian distribution, that is, zero mean and unit variance. For each catchment, it will be assigned a d-dimension vectors. d is detemrined empirically as the one yielding the best performance. 
For EA-LSTM and FM-LSTM, d is 256 or 512. For CT-LSTM, d is 16 or 32. 

#### Content of the Repository

#### Run the local Code
Train and test model. For ease of both training and testing, we put these 2 procedures together in one run. The periods of training and testing are specified in main.py. You'll need to run the following line of code from the terminal. 
```
python python main.py train --camels_root /path/to/camels --cache_data True --num_workers 12 --cluster NUMBER --attri_rand_seed RANDOM --rand_feat_num FEAT
```
A single LSTM (EA or CT) will be trained with a specified d Gaussian dimension on a specified set of basins (either subset of 531 basins or the total 531 basins).
- ```--camels-root``` directory leading to the camels dataset in your project. 
- ```--cache_data```
- ```--num_workers```
- - ```--cluster```
- ```--attri_rand_seed```
- ```--rand_feat_num```

