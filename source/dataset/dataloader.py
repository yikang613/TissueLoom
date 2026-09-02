import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import numpy as np
import torch.nn.functional as F


def init_dataloader(cfg: DictConfig,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    dataset = utils.TensorDataset(
        final_timeseires[:train_length+val_length+test_length],
        final_pearson[:train_length+val_length+test_length],
        labels[:train_length+val_length+test_length]
    )

    train_dataset, val_dataset, test_dataset = utils.random_split(
        dataset, [train_length, val_length, test_length])
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array = None) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    if stratified is None:
        stratified = labels[:, 1].cpu().numpy().astype(int)
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_kfold_dataloader(cfg: DictConfig,
                                     final_timeseires: torch.tensor,
                                     final_pearson: torch.tensor,
                                     labels: torch.tensor,
                                     stratified: np.array = None) -> List[utils.DataLoader]:
    labels_int = labels.to(torch.int64)
    labels_one_hot = F.one_hot(labels_int)

    if stratified is None:
        stratified = labels_int.cpu().numpy().astype(int)
    else:
        stratified = np.asarray(stratified)

    length = final_timeseires.shape[0]
    percentage = float(cfg.datasz.percentage)
    split_seed = int(getattr(
        cfg.dataset.k_fold,
        'split_seed',
        getattr(cfg.dataset.k_fold, 'random_state', 42),
    ))
    inner_split_seed = int(getattr(
        cfg.dataset.k_fold,
        'inner_split_seed',
        split_seed,
    ))

    if percentage < 1.0:
        subset_size = int(length * percentage)
        subset_split = StratifiedShuffleSplit(
            n_splits=1,
            train_size=subset_size,
            random_state=split_seed,
        )
        subset_indices, _ = next(subset_split.split(np.zeros(length), stratified))
        final_timeseires = final_timeseires[subset_indices]
        final_pearson = final_pearson[subset_indices]
        labels_one_hot = labels_one_hot[subset_indices]
        stratified = stratified[subset_indices]

    n_splits = int(getattr(cfg.dataset.k_fold, 'n_splits', 5))
    fold_index = int(getattr(cfg.dataset.k_fold, 'current_fold', 0))
    shuffle = bool(getattr(cfg.dataset.k_fold, 'shuffle', True))
    random_state = split_seed
    if not shuffle:
        random_state = None

    if fold_index < 0 or fold_index >= n_splits:
        raise ValueError(f'Invalid current_fold={fold_index} for n_splits={n_splits}')

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    trainval_index, test_index = list(skf.split(np.zeros(len(stratified)), stratified))[fold_index]

    val_ratio = float(cfg.dataset.val_set) / float(cfg.dataset.train_set + cfg.dataset.val_set)
    trainval_stratified = stratified[trainval_index]
    split_tv = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio,
        random_state=inner_split_seed,
    )
    inner_train_rel, inner_val_rel = next(
        split_tv.split(np.zeros(len(trainval_index)), trainval_stratified)
    )

    train_index = trainval_index[inner_train_rel]
    val_index = trainval_index[inner_val_rel]

    with open_dict(cfg):
        cfg.steps_per_epoch = (len(train_index) - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    train_dataset = utils.TensorDataset(
        final_timeseires[train_index],
        final_pearson[train_index],
        labels_one_hot[train_index],
    )

    val_dataset = utils.TensorDataset(
        final_timeseires[val_index],
        final_pearson[val_index],
        labels_one_hot[val_index],
    )

    test_dataset = utils.TensorDataset(
        final_timeseires[test_index],
        final_pearson[test_index],
        labels_one_hot[test_index],
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
