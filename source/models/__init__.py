from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .fbnetgen import FBNETGEN
from .BNT import BrainNetworkTransformer
from .tissueformer.tissue_aware_bnt_v2 import TissueAwareBNT, TissueAwareBNT_Concat
from .tissueformer.ta_bnt_v2_refined import TissueAwareBNT_v2
from .tissueformer.ta_bnt_v3 import TissueAwareBNT_v3
from .tissueformer.ta_bnt_final import TissueAwareBNT_final
from .brainnetmlp import BrainNetMLP
from .DHGFormer import DHGFormer
from .LRBGT import LRBGT
from .ComBrainTF import ComBrainTF


def _resolve_model_name(model_cfg: DictConfig):
    if hasattr(model_cfg, 'name'):
        return model_cfg.name

    selected_experiment = None
    if hasattr(model_cfg, 'experiment'):
        selected_experiment = model_cfg.experiment
    elif hasattr(model_cfg, 'exp_name'):
        selected_experiment = model_cfg.exp_name

    if selected_experiment and hasattr(model_cfg, selected_experiment):
        selected_cfg = getattr(model_cfg, selected_experiment)
        if hasattr(selected_cfg, 'name'):
            return selected_cfg.name

    raise ValueError(
        "Could not resolve model name from config. "
        "Set model.name directly, or set model.experiment/model.exp_name "
        "to one of the experiment blocks that contains a name field."
    )


def model_factory(config: DictConfig):
    model_name = _resolve_model_name(config.model)
    if model_name in ["LogisticRegression", "SVC"]:
        return None
    return eval(model_name)(config).cuda()
