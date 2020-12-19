# Tools to optimize hyperparameters of metric learning

## Install

```
pip install git+https://github.com/kosuke1701/optuna-metric-learning.git
```

## Usage
### Hyperparameter optimization tool

Run `python -m optuna_metric_learning` with appropriate arguments (see below).
`--conf` and `--model-def-fn` arguments are always required.

```
$ python -m optuna_metric_learning --help

usage: __main__.py [-h] --conf CONF --model-def-fn MODEL_DEF_FN
                   [--max-epoch MAX_EPOCH] [--patience PATIENCE]
                   [--n-trials N_TRIALS] [--sampler {Default,Random}]
                   [--log-dir LOG_DIR] [--db-name DB_NAME]
                   [--study-name STUDY_NAME]

Optimize hyperparameters of metric learning using optuna.

optional arguments:
  -h, --help            show this help message and exit
  --conf CONF           Configuration file. (default: None)
  --model-def-fn MODEL_DEF_FN
                        Model definition file. (default: None)
  --max-epoch MAX_EPOCH
                        Maximum number of epochs per trial. (default: 40)
  --patience PATIENCE   Stop training if `epoch - best_epoch > patience`.
                        (default: 1)
  --n-trials N_TRIALS   Number of trials. (default: 100)
  --sampler {Default,Random}
                        Optuna sampler. (default: None)
  --log-dir LOG_DIR     Directory name to save logging information. (default:
                        ./optuna_metric_learning)
  --db-name DB_NAME     Database to store results of each trials. If None,
                        sqlite3 database file,`optuna.sqlite3` will be created
                        at --log-dir. (default: None)
  --study-name STUDY_NAME
                        Study name. (default: metric_learning)
```

#### Model definition file

See [example directory](examples) to see examples of a model definition file and its configuration file.

"Model definition file" is a python script which defines a function `get()`. This function is called with arguments, `conf` and `trial`, and returns a set of functions which generates:

* embedding model,
* metric learning loss,
* optimizer,
* dataset.

Arguments of `get(conf, trial)` function are:

* `conf`
  - a dictionary which is loaded from JSON file which is specified by `--conf` option.
* `trial`
  - a trial object of optuna library. See [optuna documentation](https://optuna.readthedocs.io/en/stable/reference/trial.html) for more details.

Return value of `get(conf, trial)` function is:

* a dictionary, `{"modules": model_fn, "fold_generator": dataset_fn}`.
  - `model_fn`
    - a function which returns `(model_dict, optim_dict, loss_dict)`. No arguments is passed to the function.
      - Each returned dictionaries are passed to Trainer instance of pytorch metric learning as `models`, `optimizers`, `loss_funcs`. See [pytorch metric learning documentation](https://kevinmusgrave.github.io/pytorch-metric-learning/trainers/) for more details.
  - `dataset_fn`
    - This function is a generator which yields `(train_dataset, dev_dataset, train_sampler, batch_size)`.
      - `train_dataset`, `dev_dataset` are PyTorch Dataset instance.
      - `train_sampler` is a sampler which is passed to Trainer instance of pytorch metric learning.
      - `batch_size` is a batch size during training.
