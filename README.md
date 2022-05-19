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
Running the experiments listed below will reproduce the results from the paper. 
|Model architecture | 27-d Physics descriptors | Gaussian vectors (d-dimensional) | mixed Gaussian (d-dimensional) vectors | one hot vector |
|-------------------| ------------------------ | -------------------------------- | -------------------------------------- | -------------- |
|EA|||||
|CT|||||
|SR|||||
|FM|||||
# Results struture
