from argparse import ArgumentParser
from importlib import import_module
import json
import os
import sys

import numpy as np
import optuna
from pytorch_metric_learning import trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from radam import RAdam

parser = ArgumentParser()

parser.add_argument("--conf", type=str, required=True)
parser.add_argument("--model-def-fn", type=str, required=True)

parser.add_argument("--max-epoch", type=int, default=40)
parser.add_argument("--patience", type=int, default=1)
parser.add_argument("--n-trials", type=int, default=100)

parser.add_argument("--log-dir", type=str, default="./optuna_metric_learning")
parser.add_argument("--study-name", type=str, default="metric_learning")

args = parser.parse_args()

# Load model, optimizer & dataset constructor
with open(args.conf) as h:
    CONF = json.load(h)

sys.path.append(os.path.dirname(args.model_def_fn))
MODEL_DEF = import_module(os.path.basename(args.model_def_fn)[:-3])# remove ".py"

def objective(trial):
    # Average results of multiple folds.
    print("New parameter.")
    metrics = []
    constructors = MODEL_DEF.get(CONF, trial)
    for i_fold, (train_dataset, dev_dataset, train_sampler, batch_size) in enumerate(constructors["fold_generator"]()):
        print(f"Fold {i_fold}")
        model, optimizer, loss = constructors["modules"]()

        # logging
        record_keeper, _, _ = logging_presets.get_record_keeper(
            csv_folder=os.path.join(args.log_dir, f"trial_{trial.number}_{i_fold}_csv"),
            tensorboard_folder=os.path.join(args.log_dir, f"trial_{trial.number}_{i_fold}_tensorboard")
        )
        hooks = logging_presets.get_hook_container(record_keeper)

        # tester
        tester = testers.GlobalEmbeddingSpaceTester(
            end_of_testing_hook=hooks.end_of_testing_hook,
            dataloader_num_workers=32
        )
        end_of_epoch_hook = hooks.end_of_epoch_hook(
            tester, {"val": dev_dataset},
            os.path.join(args.log_dir, f"trial_{trial.number}_{i_fold}_model"),
            test_interval=1,
            patience=args.patience
        )

        # train
        trainer = trainers.MetricLossOnly(
            model, optimizer, batch_size, loss, 
            mining_funcs={}, dataset=train_dataset,
            sampler=train_sampler, dataloader_num_workers=32,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook
        )
        trainer.train(num_epochs=args.max_epoch)

        rslt = hooks.get_accuracy_history(tester, "val", metrics=["mean_average_precision_at_r"])
        
        metrics.append(max(rslt["mean_average_precision_at_r_level0"]))
    return np.mean(metrics)


os.makedirs(args.log_dir, exist_ok=True)
_storage_fn = f"sqlite:///{args.log_dir}/optuna.sqlite3".replace("/", os.path.sep)
study = optuna.create_study(
    study_name=args.study_name,
    storage=_storage_fn,
    load_if_exists=True,
    direction="maximize"
)
study.optimize(objective, n_trials=args.n_trials)
