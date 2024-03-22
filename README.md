# Usage

## Installation
First set your working directory to root project directory. Then 
install the requirements by running the following command in the terminal:

```
pip install . --extra-index-url https://download.pytorch.org/whl/cu121
```

## Running
Ensure the config file, training data and testing data are in the following paths:

```
data\raw\train.json
data\raw\test.json
data\config.yaml
```

When using other paths, specify them using flags when running the main script (For a list of flags and their 
descriptions, run the main script with a '-h' flag):

```
python .\src\main.py --config <path-to-config> --train-data <path-to-train> --test-data <path-to-test>
```


The main script expects a configuration (set of hyperparameters) from the config file, or their tuning information when 
the `--tune` flag is used. Loss curves are stored under the data folder, with the exact subfolder depending on the 
argument of the --run-id flag (which has a default value based on the time and date).  

For tuning, a different config file is required with structure as seen in [hp_tuning.yaml](data/hp_tuning.yaml). The 
path to this file should be provided as argument to the `--config` flag when tuning hyperparameters.

## Tensorboard
To view metrics while the model is being trained, open the localhost URL printed in the terminal when starting the 
program.

## Optuna dashboard
For viewing the tuning progress and results, find the `tune.db` file  in `data\results\<run-id>` when running the model 
with the `--tune` flag, and open it in optuna dashboard. For a full guide of how to use dashboard, refer to the 
[github page](https://github.com/optuna/optuna-dashboard). To use (a limited version of) optuna dashboard without 
installing any other requirements, we recommend using the 
[Browser only version](https://optuna.github.io/optuna-dashboard/). 