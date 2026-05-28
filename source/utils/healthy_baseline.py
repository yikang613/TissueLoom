"""
Healthy Baseline Utility for Normative Bipartite Projection (Exp 14/15)

Computes mean and std of WM-GM biadjacency from training CN subjects.
Must be called at the start of each CV fold, BEFORE training begins.

Usage:
    from utils.healthy_baseline import setup_healthy_baseline

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        model = TissueAwareBNT_v2(config).to(device)

        # === REQUIRED for Exp 14/15: set baseline before training ===
        setup_healthy_baseline(model, dataset, train_idx, labels, device)

        # ... train and evaluate as usual ...
"""

import torch
import numpy as np


"""
Healthy Baseline & Functional Prior Utilities for TA-BNT v2

Two functions:
1. setup_healthy_baseline_from_dataloader: For Exp 14/15 deviation modules
2. setup_functional_prior_from_dataloader: For Experiment B soft prior

Both compute statistics from CN training subjects only, per CV fold.
"""

import torch
import numpy as np


def setup_healthy_baseline_from_dataloader(model, train_dataloader, device='cuda'):
    """
    Compute and set healthy baseline from CN subjects in training DataLoader.
    Required for coupling_mode='bipartite_deviation' or 'bipartite_deviation_only'.

    Args:
        model: TissueAwareBNT_v2 instance
        train_dataloader: training DataLoader (batches of (time_series, node_feature, label))
                          where label is one-hot [B, 2]
        device: torch device string

    Returns:
        B_mean, B_std or None, None if model doesn't need baseline
    """
    coupling = getattr(model, 'coupling_module', None)
    if coupling is None or not hasattr(coupling, 'set_healthy_baseline'):
        return None, None

    num_gm = getattr(model, 'num_gm_nodes', 200)
    num_wm = getattr(model, 'num_wm_nodes', 48)

    biadj_list = []
    for time_series, node_feature, label in train_dataloader:
        cn_mask = (label[:, 1] == 0)
        if cn_mask.any():
            cn_features = node_feature[cn_mask]
            biadj = cn_features[:, :num_gm, num_gm:num_gm + num_wm]
            biadj_list.append(biadj)

    if len(biadj_list) == 0:
        raise ValueError("No CN subjects found in training dataloader.")

    B_all = torch.cat(biadj_list, dim=0)
    if B_all.shape[0] < 2:
        raise ValueError(f"Need >= 2 CN subjects, found {B_all.shape[0]}.")

    B_mean = B_all.mean(dim=0)
    B_std = B_all.std(dim=0).clamp(min=1e-6)

    coupling.set_healthy_baseline(B_mean.to(device), B_std.to(device))

    print(f"[Healthy Baseline] Set from {B_all.shape[0]} CN training subjects")
    print(f"  B_mean: [{B_mean.min():.4f}, {B_mean.max():.4f}]")
    print(f"  B_std:  [{B_std.min():.4f}, {B_std.max():.4f}]")

    return B_mean, B_std


def setup_functional_prior_from_dataloader(model, train_dataloader, device='cuda'):
    """
    Compute and set functional prior from CN subjects in training DataLoader.
    Required when model config has use_prior=True.

    The prior is the mean |FC| across CN training subjects, normalized to [0,1].
    It biases the biadjacency toward the healthy population pattern.

    Args:
        model: TissueAwareBNT_v2 instance with BipartiteProjectionCombined(use_prior=True)
        train_dataloader: training DataLoader
        device: torch device string

    Returns:
        B_prior [num_gm, num_wm] or None if model doesn't use prior
    """
    coupling = getattr(model, 'coupling_module', None)
    if coupling is None or not hasattr(coupling, 'set_functional_prior'):
        return None

    if not getattr(coupling, 'use_prior', False):
        return None

    num_gm = getattr(model, 'num_gm_nodes', 200)
    num_wm = getattr(model, 'num_wm_nodes', 48)

    biadj_list = []
    for time_series, node_feature, label in train_dataloader:
        cn_mask = (label[:, 1] == 0)
        if cn_mask.any():
            cn_features = node_feature[cn_mask]
            biadj = cn_features[:, :num_gm, num_gm:num_gm + num_wm]
            biadj_list.append(biadj)

    if len(biadj_list) == 0:
        raise ValueError("No CN subjects found in training dataloader.")

    B_all = torch.cat(biadj_list, dim=0)  # [N_cn, 200, 48]

    # Mean absolute FC as prior (absolute because we care about coupling
    # magnitude regardless of sign)
    B_prior = B_all.abs().mean(dim=0)  # [200, 48]

    coupling.set_functional_prior(B_prior.to(device))

    print(f"[Functional Prior] Set from {B_all.shape[0]} CN training subjects")
    print(f"  B_prior: [{B_prior.min():.4f}, {B_prior.max():.4f}], "
          f"mean={B_prior.mean():.4f}")
    prior_nonzero = (B_prior > 0.01).sum().item()
    print(f"  Pairs with |FC| > 0.01: {prior_nonzero}/{B_prior.numel()} "
          f"({100*prior_nonzero/B_prior.numel():.1f}%)")

    return B_prior


def verify_baseline_set(model):
    """Check whether required baselines/priors are set."""
    coupling = getattr(model, 'coupling_module', None)
    if coupling is None:
        return True

    coupling_mode = getattr(model, 'coupling_mode', 'none')

    # Check deviation baseline
    needs_baseline = coupling_mode in ('bipartite_deviation', 'bipartite_deviation_only')
    if needs_baseline:
        is_set = getattr(coupling, '_baseline_set', False)
        if not is_set:
            print(f"WARNING: coupling_mode='{coupling_mode}' requires "
                  f"set_healthy_baseline() but baseline is not set!")
            return False

    # Check functional prior
    uses_prior = getattr(coupling, 'use_prior', False)
    if uses_prior:
        is_set = getattr(coupling, '_prior_set', False)
        if not is_set:
            print(f"WARNING: use_prior=True but functional prior is not set!")
            return False

    return True


def verify_baseline_set(model):
    """
    Check whether the healthy baseline has been set in the model.
    Call this before training to catch configuration errors early.

    Args:
        model: TissueAwareBNT_v2 instance

    Returns:
        True if baseline is set, False otherwise

    Raises:
        Warning message if coupling mode requires baseline but it's not set
    """
    coupling = getattr(model, 'coupling_module', None)
    if coupling is None:
        return True  # No coupling module, no baseline needed

    coupling_mode = getattr(model, 'coupling_mode', 'none')
    needs_baseline = coupling_mode in (
        'bipartite_deviation', 'bipartite_deviation_only'
    )

    if not needs_baseline:
        return True  # This coupling mode doesn't need a baseline

    is_set = getattr(coupling, '_baseline_set', False)
    if not is_set:
        print(f"WARNING: coupling_mode='{coupling_mode}' requires "
              f"set_healthy_baseline() but baseline is not set! "
              f"Call setup_healthy_baseline() before training.")
    return is_set
