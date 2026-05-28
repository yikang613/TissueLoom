from operator import mod
from .training import Train
from .FBNettraining import FBNetTrain
from omegaconf import DictConfig
from typing import List
import torch
from source.components import LRScheduler
import logging
import torch.utils.data as utils

from .tabnt_training import TABNTTrain


def _resolve_train_name(config: DictConfig) -> str:
    """Resolve training class name from nested experiment config or fallback."""
    # 0. Explicit override from training config (highest priority)
    #    Example: training.train=TABNTTrain or training.train=Train
    train = config.training.get("train", None)
    if train:
        return train

    # 1. Check top-level model.train
    train = config.model.get("train", None)
    if train:
        return train

    # 2. Check inside the resolved experiment block (e.g., model.t1_class_weight.train)
    selected = getattr(config.model, 'experiment', None) or getattr(config.model, 'exp_name', None)
    if selected and hasattr(config.model, selected):
        exp_cfg = getattr(config.model, selected)
        train = getattr(exp_cfg, 'train', None)
        if train:
            return train

    # 3. Fallback to training config
    return config.training.name


def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger) -> Train:

    train = _resolve_train_name(config)
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger)
