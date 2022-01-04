# Global-deep-learning-regionalization-from-physical-descriptors-to-random-vectors
The code for performing global deep learning regionalization from physical descriptors to random vectors. The paper is in reviewing processes. 

The code for reproducing the workflow is being organized and will be published soon. The scripts are largely based on Kratzert's work on Catchment-Aware LSTMs for Regional Rainfall-Runoff Modeling (https://github.com/kratzert/ealstm_regional_modeling). Minor revisions regarding how to assign random vectors to catchments, which are proposed solutions in the paper, are made and thus are shown in this repo. 

This paper highlights the benefit of using random vector as a substitute strategy for hydrologic global regionalization when catchment static catchment characteristics are uncertain, incompelte, and unavailable. 

For replicating the LSTM based models using the static features, please run the original script shared by Kratzert in his github repository (https://github.com/kratzert/ealstm_regional_modeling). 

"Gaussian" folder and "one_hot" folders are basically the orginal EA-LSTM model repositories, the revision is committed for workflow and static feature generations.
