# Global-deep-learning-regionalization-from-physical-descriptors-to-random-vectors
The code for performing global deep learning regionalization from physical descriptors to random vectors. The paper is in reviewing processes. 

The code for reproducing the workflow is being organized and will be published soon. The scripts are largely based on Kratzert's work on Catchment-Aware LSTMs for Regional Rainfall-Runoff Modeling (https://github.com/kratzert/ealstm_regional_modeling). Minor revisions regarding how to assign random vectors to catchments, which are proposed solutions in the paper, are made and thus are shown in this repo. 

This paper highlights the benefit of using random vector as a substitute strategy for hydrologic global regionalization when catchment static catchment characteristics are uncertain, incompelte, and unavailable. 

For replicating the LSTM based models using the static features, please run the original script shared by Kratzert in his github repository (https://github.com/kratzert/ealstm_regional_modeling). 

About the model environment set-up and data donwloading, please check the https://github.com/kratzert/ealstm_regional_modeling and use the Updated Maurer forcing data to replicate the results from this paper and we will not reiterate anymore. 

# Content of the Repository
- data/
  - text files that store the basin IDs. The name of those text files will be arugments that pass into ```--cluster``` arguments. 
- papercode/
  - consains the entire code what main.py will load from. 
# Results struture
