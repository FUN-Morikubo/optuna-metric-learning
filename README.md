# Tools to optimize hyperparameters of metric learning

## Install

```
pip install git+https://github.com/kosuke1701/optuna-metric-learning.git
```

## Usage

Example usage of this tool:

```
python -m optuna_metric_learning --conf examples/image_folder_examples.json --model-def-fn examples/image_folder_examples.py --max-epoch 30 --log-dir test --db-name sqlite:///test.sql --n-trial 30
```

Detail configurations of models, losses, data, and data augmentation are described in `examples/image_folder_examples.json`.

### Dataset

By default, this tool load your dataset with `ImageFolder`, so please arrange your dataset as described in [documentation](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder).