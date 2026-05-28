from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .pAD import load_pAD_data
from .AD import load_AD_data
from .MCI import load_MCI_data
from .EMCI import load_EMCI_data
from .LMCI import load_LMCI_data
from .dataloader import init_dataloader, init_stratified_dataloader, init_stratified_kfold_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide', 'pAD', 'AD', 'MCI', 'EMCI', 'LMCI'], f"Unsupported dataset: {cfg.dataset.name}"

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    kfold_enabled = bool(getattr(getattr(cfg.dataset, 'k_fold', {}), 'enabled', False))

    if kfold_enabled:
        dataloaders = init_stratified_kfold_dataloader(cfg, *datasets)
    elif cfg.dataset.stratified:
        dataloaders = init_stratified_dataloader(cfg, *datasets)
    else:
        dataloaders = init_dataloader(cfg, *datasets)

    return dataloaders
