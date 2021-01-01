from argparse import ArgumentParser
from glob import glob
from importlib import import_module
import json
import os
import shutil
import sys

import optuna
from pytorch_metric_learning import trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets

parser = ArgumentParser()

parser.add_argument("--conf", type=str, required=True, help="Configuration file.")
parser.add_argument("--model-def-fn", type=str, required=True, help="Model definition file.")

parser.add_argument("--log-dir", type=str, default="./optuna_metric_learning", help="Directory name to save logging information.")
parser.add_argument("--db-name", type=str, required=True, 
    help="Database to store results of each trials. If None, sqlite3 database file,`optuna.sqlite3` will be created at --log-dir.")
parser.add_argument("--study-name", type=str, default="metric_learning", help="Study name.")
parser.add_argument("--model-save-dir", type=str, required=True)

parser.add_argument("--max-epoch", type=int, default=30)
parser.add_argument("--patience", type=int, default=1)

parser.add_argument("--n-fold", type=int, default=10)

parser.add_argument("--trial", type=int, default=-1)

args = parser.parse_args()

# Load model, optimizer & dataset constructor
with open(args.conf) as h:
    CONF = json.load(h)
CONF["n_fold"] = args.n_fold
CONF["fold_trials"] = 1

sys.path.append(os.path.dirname(args.model_def_fn))
MODEL_DEF = import_module(os.path.basename(args.model_def_fn)[:-3])

# Load best parameter
study = optuna.load_study(
    study_name=args.study_name,
    storage=args.db_name
)
if args.trial < 0:
    best_trial = study.best_trial
    best_params = study.best_params
else:
    best_trial = study.trials[args.trial]
    best_params = best_trial.params

print("Use following best parameter:")
print(best_params)

constructors = MODEL_DEF.get(CONF, best_trial)

train_dataset, dev_dataset, train_sampler, batch_size = \
    next(constructors["fold_generator"]())
model, optimizer, loss = constructors["modules"]()

# logging
record_keeper, _, _ = logging_presets.get_record_keeper(
    csv_folder=os.path.join(args.log_dir, f"csv"),
    tensorboard_folder=os.path.join(args.log_dir, f"tensorboard")
)
hooks = logging_presets.get_hook_container(record_keeper)

# tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    dataloader_num_workers=32
)
end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, {"val": dev_dataset},
    os.path.join(args.log_dir, f"model"),
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
print("Best MAP@R=", max(rslt["mean_average_precision_at_r_level0"]))

# Save best model
os.makedirs(args.model_save_dir, exist_ok=True)
model_dir = os.path.join(args.log_dir, f"model")

best_embedder = glob(os.path.join(model_dir, "embedder_best*.pth"))[0]
best_trunk = glob(os.path.join(model_dir, "trunk_best*.pth"))[0]

shutil.copy(best_embedder, os.path.join(args.model_save_dir, "embedder.pth"))
shutil.copy(best_trunk, os.path.join(args.model_save_dir, "trunk.pth"))
with open(os.path.join(args.model_save_dir, "conf.json"), "w") as h:
    json.dump([CONF, best_params], h)