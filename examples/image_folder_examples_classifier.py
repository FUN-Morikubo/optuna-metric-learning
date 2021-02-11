from glob import glob
import os
import re

import numpy as np
from pytorch_metric_learning import samplers
from radam import RAdam
from RandAugment import RandAugment
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import timm

from optuna_metric_learning.models import ALL_LOSSES

def get(conf, trial, param_gen):
    SIZE = conf["input_size"]

    # TRANSFORM
    trans = [
        transforms.Resize((SIZE,SIZE)),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    trans = transforms.Compose(trans)

    if conf["use_augmentation"]:
        train_trans = [
            transforms.Resize((SIZE, SIZE)),
            transforms.CenterCrop(SIZE)
        ]
        
        # Flip
        if param_gen.suggest_categorical("augment_flip", [0,1]) > 0:
            train_trans.append(transforms.RandomHorizontalFlip())
        # Rotation, Shift, Scale
        rot_degree = param_gen.suggest_uniform("augment_rot", 0.0, 180.0)
        translate = param_gen.suggest_uniform("augment_trans", 0.0, 0.3)
        scale = param_gen.suggest_uniform("augment_scale", 1.0, 1.3)
        train_trans.append(transforms.RandomAffine(
            rot_degree, translate=(translate, translate), scale=(1/scale, scale)))
        
        train_trans += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]

        # Erasing
        if param_gen.suggest_categorical("augment_erase", [0,1]) > 0:
            train_trans.append(transforms.RandomErasing())

        train_trans = transforms.Compose(train_trans)
    elif conf["use_randaug"]:
        _N = param_gen.suggest_int("randaug_N", 0, 10)
        _M = param_gen.suggest_int("randaug_M", 0, 30)
        train_trans = [
            transforms.Resize((SIZE, SIZE)),
            transforms.CenterCrop(SIZE),
            RandAugment(_N, _M),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        train_trans = transforms.Compose(train_trans)
    else:
        train_trans = trans

    trans_dict = {"train": train_trans, "valid": trans}

    # DATASET GENERATOR
    label_dirnames = glob(os.path.join(conf["data_dir"], "*"))
    labels = []
    for dirname in label_dirnames:
        label = dirname.split(os.path.sep)[-1]
        labels.append(label)
    labels = sorted(labels)
    def dataset_fold_generator(all=False):
        folds = KFold(n_splits=conf["n_fold"], shuffle=True, random_state=1234)
        dummy = np.arange(len(labels), dtype=np.int32)[:, np.newaxis]
        for train_lab, dev_lab in list(folds.split(dummy))[:conf["fold_trials"]]:
            train_lab = set([labels[_] for _ in train_lab])
            dev_lab = set([labels[_] for _ in dev_lab])

            train_dataset = ImageFolder(
                root=conf["data_dir"],
                transform=trans_dict["train"],
                is_valid_file=lambda fn: fn.split(os.path.sep)[-2] in  train_lab
            )
            train_sampler = samplers.MPerClassSampler(
                labels=train_dataset.targets,
                m = param_gen.suggest_categorical("m", conf["m_cands"]),
                batch_size=conf["batch_size"],
                length_before_new_iter=conf["data_per_epoch"]
            )
            dev_dataset = ImageFolder(
                root=conf["data_dir"],
                transform=trans_dict["valid"],
                is_valid_file=lambda fn: fn.split(os.path.sep)[-2] in  dev_lab
            )

            yield train_dataset, dev_dataset, train_sampler, conf["batch_size"]
    
    # MODEL, OPTIMIZER & LOSS
    def model_generator():
        if conf["trunk_model"] in ["ResNet-18", "ResNet-34", "ResNet-50", "ResNet-101", "ResNet-152"]:
            depth = re.match("ResNet-(?P<depth>[0-9]+)", conf["trunk_model"]).group("depth")
            trunk = eval(f"models.resnet{depth}(pretrained=True)")
            trunk_output_size = trunk.fc.in_features
            trunk.fc = nn.Identity()
        elif conf["trunk_model"].startswith("timm:"):
            _model_name = conf["trunk_model"][5:]
            trunk = timm.create_model(_model_name, pretrained=True)
            trunk.reset_classifier(0)
            trunk_output_size = trunk.num_features

        p_dropout = param_gen.suggest_uniform("p_dropout", 0.0, 1.0)
        embedder = nn.Sequential(
            nn.Linear(trunk_output_size, conf["dim"]),
            nn.Dropout(p_dropout)
        )

        classifier = nn.Linear(conf["dim"], len(labels))

        trunk.to("cuda")
        embedder.to("cuda")
        classifier.to("cuda")

        model_dict = {"trunk": trunk, "embedder": embedder, "classifier": classifier}

        #lr = param_gen.suggest_loguniform("model_lr", 1e-5, 1e-3)
        # beta1 = 0.9
        # beta2 = 0.999
        # eps = 1e-8
        decay = param_gen.suggest_loguniform("model_decay", 1e-10, 1e-2)
        lr = param_gen.suggest_loguniform("model_lr", 1e-5, 1e-4)
        beta1 = 1. - param_gen.suggest_loguniform("model_beta1", 1e-3, 1.)
        beta2 = 1. - param_gen.suggest_loguniform("model_beta2", 1e-4, 1.)
        eps = param_gen.suggest_loguniform("model_eps", 1e-8, 1)
        optim_dict = {
            "trunk_optimizer": RAdam(trunk.parameters(), 
                    lr=lr, 
                    weight_decay=decay,
                    betas=(beta1, beta2),
                    eps=eps
                    ),
            "embedder_optimizer": RAdam(embedder.parameters(), 
                    lr=lr, 
                    weight_decay=decay,
                    betas=(beta1, beta2),
                    eps=eps
                    ),
            "classifier_optimizer": RAdam(classifier.parameters(),
                    lr=lr, 
                    weight_decay=decay,
                    betas=(beta1, beta2),
                    eps=eps
                )
        }

        loss_type = conf["loss_type"]
        metric_loss_info = ALL_LOSSES[loss_type](param_gen,
            num_classes=len(labels), embedding_size=conf["dim"])
        if "param" in metric_loss_info:
            # Add optimizer for loss
            lr = param_gen.suggest_loguniform("loss_lr", 1e-5, 1e-2)
            # beta1 = 1. - param_gen.suggest_loguniform("loss_beta1", 1e-3, 1.)
            # beta2 = 1. - param_gen.suggest_loguniform("loss_beta2", 1e-4, 1.)
            # eps = param_gen.suggest_loguniform("loss_eps", 1e-10, 1e-5)
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            opt = RAdam(metric_loss_info["loss"].parameters(),
                lr=lr, betas=(beta1, beta2), eps=eps)
            optim_dict["metric_loss_optimizer"] = opt
        loss_dict = {
            "metric_loss": metric_loss_info["loss"],
            "classifier_loss": torch.nn.CrossEntropyLoss()
        }

        cls_weight = param_gen.suggest_loguniform("cls_weight", 0.01, 100.0)
        loss_weights = {"metric_loss": 1, "classifier_loss": cls_weight}

        return {
            "models": model_dict,
            "optimizers": optim_dict,
            "loss_funcs": loss_dict,
            "loss_weights": loss_weights
        }
        
    return {"modules": model_generator, "fold_generator": dataset_fold_generator}