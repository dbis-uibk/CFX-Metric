# CFX-Metric
Fair Metric for Counterfactual Explanation of Recommenders


## Repository

This repository contains code for "Are We Explaining Flawed Recommenders? Incorporating Recommender Performance in evaluating explainers" paper evaluated on three publicly available benchmarks, MovieLens1M, a subset of Yahoo!Music dataset and a subset of Pinterest dataset, using two different recommenders, Matric Factorization (MF) and Variational Auto Encoder (VAE). Hyperparameters optimization was done using optuna.

## Folders

* **processed_data**: contains three subfolders, one for each dataset. In each dataset folder lies the raw data files.
* **code**: contains several code notebooks:
  - data_processing - code related to the preprocessing step for preparing data to run with our models.
  - help_functions - includes the framework's functions that are being used in all notebooks.
  - recommenders_architecture - specifies the architecture of the recommenders that were used in the paper.
  - recommenders_training - contains code related to VAE and MLP recommenders training.
  - LXR_training - contains code for training LXR model for explaining a specified recommender.
  - metrics - contains code related to model evaluation.
* **baselines**: contains code for LIME and SHAP baselines.
* **checkpoints**: presently, this folder is empty. It is the designated location for saving and loading the trained model's checkpoints. The checkpoints developed during our project are stored in the 'checkpoints' folder on the [drive](https://drive.google.com/drive/u/3/folders/1nD0_5asi4B9dyUN_JYoYT5QJPYeAMWCD).
  
## Requirements

* python 3.10
* Pytorch 1.13
* wandb 0.16.3 (the package we used for monitoring the train process)

## Usage

To use this code, follow these steps:
+ Create data to work with by running the data_processing notebooks.
  - Or in order to reproduce results from the paper without running the data_processing notebook, please download all files from [here](https://drive.google.com/drive/folders/1nD0_5asi4B9dyUN_JYoYT5QJPYeAMWCD?usp=sharing) from the relevant folder <dataset_name> to data_preprocessing folder according to the data set you need to run on. 
+ On every notebook, please specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE'.
+ You can train your oun LXR with the 'LXR_training' notebook, test your results in 'metrics' notebook, and test new explanation methods using 'metrics' notebook. 


![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/MLP%20ML-1M.png)


![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/VAE%20ML1M.png)


![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/VAE%20Pinterest.png)


![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/Yahoo%20MLP.png)
