# Tools to optimize hyperparameters of metric learning

## Install

```
pip install git+https://github.com/kosuke1701/optuna-metric-learning.git
```

## Usage

### Hyperparameter tuning

```
python -m optuna_metric_learning --conf examples/image_folder_examples.json --model-def-fn examples/image_folder_examples.py --max-epoch 30 --log-dir train --db-name sqlite:///tuning.sql --n-trial 30
```

Detail configurations of models, losses, data, and data augmentation are described in `examples/image_folder_examples.json`.

### Training with tuned hyperparameters

By default, hyperparameter of the best trial is used for training. Use same `--conf` and `--model-def-fn` as in hyperparameter tuning.

```
python -m optuna_metric_learning.train --conf examples/image_folder_example.json --model-def-fn examples/image_folder_example.py --log-dir test --db-name sqlite:///tuning.sql --model-save-dir trained_model
```

### Dataset

By default, this tool load your dataset with `ImageFolder`, so please arrange your dataset as described in [documentation](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).