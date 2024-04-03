# Question Answering by fine-tuning pretrained models.
[![Python](https://img.shields.io/badge/python-3.9%20-blue)](https://www.python.org)

This project contains the implementation for fine-tuning pretrained models for question answering with the 
[SQuAD v2.0 dataset](https://rajpurkar.github.io/SQuAD-explorer/). While not producing SOTA performance, this approach 
is an accesible method for achieving high performance without access to powerful computational resources or days of 
training. Even without extensive hyperparameter tuning, this approach has been found to result in high performance reliably. 

# Usage

## Installation
First set your working directory to root project directory. Then 
install the requirements by running the following command in the terminal:

```
pip install .
```

Depending on your device, you may have to install a different version of PyTorch if you wish to run the code using a 
GPU. A guide to the installation process can be found [here](https://pytorch.org/get-started/locally/). 

## Running
Ensure the config files, training data and testing data are in the following paths:

```
data\raw\train.json
data\raw\test.json
data\config.yaml
data\hp_tuning.yaml
```

When using other paths or files, specify them using flags when running the main script (For a list of flags and their 
descriptions, run the main script with a '-h' flag):

```
python .\src\main.py --config <path-to-config> --train-data <path-to-train> --test-data <path-to-test>
```


The main script expects a configuration (set of hyperparameters) from the config file, or their tuning information when 
the `--tune` flag is used. Loss curves are stored under the data folder, with the exact subfolder depending on the 
argument of the --run-id flag (which has a default value based on the time and date).  

For tuning, a different config file is required with structure as seen in [hp_tuning.yaml](data/hp_tuning.yaml). The 
path to this file should be provided as argument to the `--config` flag when tuning hyperparameters.

## Optuna dashboard
For viewing the tuning progress and results, find the `tuning.db` file  in `data\results\<run-id>` when running the model 
with the `--tune` flag, and open it in optuna dashboard. For a full guide of how to use dashboard, refer to the 
[github page](https://github.com/optuna/optuna-dashboard). To use (a limited version of) optuna dashboard without 
installing any other requirements, we recommend using the 
[Browser only version](https://optuna.github.io/optuna-dashboard/), in which you can simply manually open the *.db file. 