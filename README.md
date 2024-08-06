# CFX-Metric
Do we have fair metrics for Counterfactual Explanation of Recommenders?
Are we explaining flawed recommenders and don't report the recommender performance?


## Repository

This repository contains code for "Are We Explaining Flawed Recommenders? Incorporating Recommender Performance in evaluating explainers" paper. We have our evaluated our claim on three publicly available benchmarks, MovieLens1M, a subset of Yahoo!Music dataset and a subset of Pinterest dataset, using two different recommenders, Matric Factorization (MF) and Variational Auto Encoder (VAE). Hyperparameters optimization was done using optuna.

## Folders

* **Experiments Results**: contains all the results of recommenders we used for the tables and figures in the paper
* **code**: contains several code files:
  - data_processing - code related to the preprocessing step for preparing data to run with our models.
  - help_functions - includes the framework's functions that are being used in all codes.
  - recommenders_architecture - specifies the architecture of the recommenders that were used in the paper.
  - recommenders_training - contains code related to VAE and MLP recommenders training.
  - LXR_training - contains code for training LXR model for explaining a specified recommender.
  - metrics - contains code related to model evaluation.
* **checkpoints**: It is the designated location for saving and loading the trained model's checkpoints.
  
## Requirements

* python 3.10
* Pytorch 1.13
* wandb 0.16.3 (the package we used for monitoring the train process)

## Usage

To use this code, follow these steps:
+ Create data to work with by running the data_processing code.
+ On every code, please specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE'.

## Reproducing the Results:
+ After running the preprocessing step, simply run the recommenders_training.py and specify the "data_name" variable to be 'ML1M'/'Yahoo'/'Pinterest', and the "recommender_name" variable to be 'MLP'/'VAE'.
+ From the output checkpoints check which recommenders you want to pick for explanation. Then set the file name of the checkpoint in LXR_training.py or pass it as a argument by --directory and run to train the explainers. 
+ Then to get other explainers and evaluate LXR evaluation, run the metrics.py file. This will print all the numbers you want. We have all these outputs in "Experiments Results" folder.

## Other Tables of Resutls

![MLP_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/MLP%20ML-1M.png)


![VAE_ML1M_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/VAE%20ML1M.png)


![MLP_Pinterest_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/VAE%20Pinterest.png)


![MLP_Yahoo_table](https://github.com/dbis-uibk/CFX-Metric/blob/main/Experiments%20Result/img/Yahoo%20MLP.png)
