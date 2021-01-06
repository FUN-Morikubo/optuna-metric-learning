from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from importlib import import_module
import json
import logging
import os
import sys
import traceback

import numpy as np
import optuna
import torch
from pytorch_metric_learning import trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from radam import RAdam

from optuna_metric_learning.misc import ParameterGenerator

parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Optimize hyperparameters of metric learning using optuna.",
    add_help=True
)

parser.add_argument("--conf", type=str, required=True, help="Configuration file.")
parser.add_argument("--model-def-fn", type=str, required=True, help="Model definition file.")

parser.add_argument("--max-epoch", type=int, default=40, help="Maximum number of epochs per trial.")
parser.add_argument("--patience", type=int, default=1, help="Stop training if `epoch - best_epoch > patience`.")
parser.add_argument("--n-trials", type=int, default=100, help="Number of trials.")

parser.add_argument("--sampler", type=str, choices=["Default", "Random", "Grid"], 
    default="Default", help="Optuna sampler.")

parser.add_argument("--log-dir", type=str, default="./optuna_metric_learning", help="Directory name to save logging information. NOTE: Use an unique name for each different optimization.")
parser.add_argument("--db-name", type=str, default=None, 
    help="Database to store results of each trials. If None, sqlite3 database file,`optuna.sqlite3` will be created at --log-dir.")
parser.add_argument("--study-name", type=str, default="metric_learning", help="Study name.")

parser.add_argument("--ignore-error", action="store_true", help="If this option is set, restart training from last epoch when any error occurs during a trial.")

parser.add_argument("--n-train-loader", type=int, default=4)
parser.add_argument("--n-test-loader", type=int, default=4)

args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)
logger = logging.getLogger()
logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, "optuna.log"), mode="w"))

optuna.logging.enable_propagation()
optuna.logging.disable_default_handler()

# Load model, optimizer & dataset constructor
with open(args.conf) as h:
    CONF = json.load(h)

sys.path.append(os.path.dirname(args.model_def_fn))
MODEL_DEF = import_module(os.path.basename(args.model_def_fn)[:-3])# remove ".py"

def objective(trial):
    param_gen = ParameterGenerator(trial, CONF["_fix_params"], logger=logger)

    # Average results of multiple folds.
    print("New parameter.")
    metrics = []
    constructors = MODEL_DEF.get(CONF, trial, param_gen)
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
            dataloader_num_workers=args.n_test_loader
        )
        end_of_epoch_hook = hooks.end_of_epoch_hook(
            tester, {"val": dev_dataset},
            os.path.join(args.log_dir, f"trial_{trial.number}_{i_fold}_model"),
            test_interval=1,
            patience=args.patience
        )

        CHECKPOINT_FN = os.path.join(args.log_dir, f"trial_{trial.number}_{i_fold}_last.pth")
        def actual_end_of_epoch_hook(trainer):
            continue_training = end_of_epoch_hook(trainer)

            torch.save(
                (
                    {k:m.state_dict() for k,m in trainer.models.items()},
                    {k:m.state_dict() for k,m in trainer.optimizers.items()},
                    {k:m.state_dict() for k,m in trainer.loss_funcs.items()},
                    trainer.epoch
                ),
                CHECKPOINT_FN
            )

            return continue_training
        

        # train
        trainer = trainers.MetricLossOnly(
            model, optimizer, batch_size, loss, 
            mining_funcs={}, dataset=train_dataset,
            sampler=train_sampler, dataloader_num_workers=args.n_train_loader,
            end_of_iteration_hook=hooks.end_of_iteration_hook,
            end_of_epoch_hook=actual_end_of_epoch_hook
        )

        while True:
            start_epoch = 1
            if os.path.exists(CHECKPOINT_FN):
                model_dicts, optimizer_dicts, loss_dicts, last_epoch = \
                    torch.load(CHECKPOINT_FN)
                for k,d in model_dicts.items():
                    trainer.models[k].load_state_dict(d)
                for k,d in optimizer_dicts.items():
                    trainer.optimizers[k].load_state_dict(d)
                for k,d in loss_dicts.items():
                    trainer.loss_funcs[k].load_state_dict(d)
                start_epoch = last_epoch + 1

                logger.critical(f"Start from old epoch: {last_epoch + 1}")
            try:
                trainer.train(num_epochs=args.max_epoch, start_epoch=start_epoch)
            except Exception as err:
                logger.critical(f"Error: {err}")
                if not args.ignore_error:
                    break
            else:
                break

        rslt = hooks.get_accuracy_history(tester, "val", metrics=["mean_average_precision_at_r"])
        
        metrics.append(max(rslt["mean_average_precision_at_r_level0"]))
    return np.mean(metrics)

if args.db_name is None:
    _storage_fn = f"sqlite:///{args.log_dir}/optuna.sqlite3".replace("/", os.path.sep)
else:
    _storage_fn = args.db_name

if args.sampler == "Default":
    sampler = None
elif args.sampler == "Random":
    sampler = optuna.samplers.RandomSampler()
elif args.sampler == "Grid":
    sampler = optuna.samplers.GridSampler(CONF["search_space"])

study = optuna.create_study(
    study_name=args.study_name,
    storage=_storage_fn,
    load_if_exists=True,
    direction="maximize",
    sampler=sampler
)

counter_trials = sum([1 for trial in study.trials if trial.state==optuna.trial.TrialState.COMPLETE])
print(f"{counter_trials} trials have been already completed.")
if counter_trials >= args.n_trials:
    print("Already finished. Exit.")
    sys.exit(0)
while True:
    try:
        study.optimize(objective, n_trials=1)
    except Exception as err:
        continue

    counter_trials = sum([1 for trial in study.trials if trial.state==optuna.trial.TrialState.COMPLETE])
    if counter_trials < args.n_trials:
        continue
    else:
        break