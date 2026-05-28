"""
Healthy Baseline & Zone Setup for TA-BNT v3
=============================================

Separate from healthy_baseline.py (v2) to preserve backward compatibility.
v2/B3 continues to work with the original healthy_baseline.py untouched.

This file provides setup_v3_for_fold() — call ONCE per CV fold BEFORE training.
It does three things:
  1. Computes B_CN_prior: mean |FC_GM-WM| from CN training subjects
  2. Generates zones via ZoneGenerator (which GM parcels → which WM tract circuits)
  3. Configures the model: setup_zones() for M3+M4, set_functional_prior() for bipartite

Usage in kfold_runner.py (add after the v2 setup block):

    if model_name == 'TissueAwareBNT_v3':
        from source.training.healthy_baseline_v3 import setup_v3_for_fold
        setup_v3_for_fold(model, train_loader, device='cuda')
"""

import torch
import numpy as np


def setup_v3_for_fold(model, train_dataloader, device='cuda',
                      zone_method='threshold', zone_threshold_std=0.5,
                      zone_k=15, zone_min_size=5, tract_groups=None):
    """One-call setup for TA-BNT v3 per CV fold.

    Computes B_CN_prior from CN training subjects, generates zones,
    and configures the model. Must be called BEFORE training starts.

    Args:
        model: TissueAwareBNT_v3 wrapper (has .model attribute)
               or raw TABNT_v3 instance
        train_dataloader: training DataLoader
            Yields (time_series, node_feature, label) per batch
            where label is one-hot [B, 2] (label[:, 1]==0 means CN)
        device: 'cuda' or 'cpu'
        zone_method: 'threshold' or 'topk' for ZoneGenerator
        zone_threshold_std: std multiplier for threshold method
        zone_k: number of GM parcels per zone for topk method
        zone_min_size: minimum GM parcels per zone (threshold fallback)
        tract_groups: custom tract groups dict, or None for JHU_TRACT_GROUPS

    Returns:
        zones: dict of {group_name: {'gm': [indices], 'wm': [indices]}}
        B_prior: [num_gm, num_wm] tensor
    """
    # Import atlas definitions and zone generator.
    # Try final model first, fall back to v3 for backward compatibility.
    try:
        from source.models.tissueformer.ta_bnt_final import (
            ZoneGenerator, JHU_TRACT_GROUPS, validate_tract_groups
        )
    except ImportError:
        from source.models.tissueformer.ta_bnt_v3 import (
            ZoneGenerator, JHU_TRACT_GROUPS, validate_tract_groups
        )

    # Handle both wrapper (TissueAwareBNT_v3) and raw (TABNT_v3) models
    inner = model.model if hasattr(model, 'model') else model

    if tract_groups is None:
        tract_groups = dict(JHU_TRACT_GROUPS)
    tract_groups = validate_tract_groups(tract_groups, inner.num_wm)

    num_gm = inner.num_gm
    num_wm = inner.num_wm

    # ---- Step 1: Compute B_CN_prior from CN training subjects ----
    # Follows v2 pattern: iterate dataloader, label[:, 1] == 0 for CN
    cross_sum = torch.zeros(num_gm, num_wm, device=device)
    cn_count = 0

    with torch.no_grad():
        for time_series, node_feature, label in train_dataloader:
            node_feature = node_feature.to(device)
            label = label.to(device)

            # One-hot labels [B, 2]: CN has label[:, 1] == 0
            cn_mask = (label[:, 1] == 0)

            if cn_mask.any():
                fc_cn = node_feature[cn_mask]
                fc_cross_cn = fc_cn[:, :num_gm, num_gm:num_gm + num_wm]
                cross_sum += fc_cross_cn.abs().sum(dim=0)
                cn_count += cn_mask.sum().item()

    if cn_count == 0:
        raise ValueError("No CN subjects found in training dataloader.")

    B_prior = cross_sum / cn_count  # [200, 48]
    B_prior_cpu = B_prior.cpu()

    print(f"[v3 Setup] B_CN_prior from {cn_count} CN training subjects")
    print(f"  B_prior: [{B_prior_cpu.min():.4f}, {B_prior_cpu.max():.4f}], "
          f"mean={B_prior_cpu.mean():.4f}")

    # ---- Step 2: Generate zones ----
    zone_gen = ZoneGenerator(
        tract_groups,
        method=zone_method,
        k=zone_k,
        threshold_std=zone_threshold_std,
        min_zone_size=zone_min_size
    )
    zones = zone_gen.generate_zones(B_prior_cpu)
    zone_gen.summary()

    # ---- Step 3: Configure model ----
    # Works for wrapper (TissueAwareBNT_final/v3) and raw (TABNT_Final/v3)
    # B_prior passed to setup_zones for prior-weighted circuit pooling
    model.setup_zones(zones, B_prior=B_prior_cpu)
    model.set_functional_prior(B_prior_cpu)

    coupling_mode = getattr(inner, 'coupling_mode', 'unknown')
    pooling_mode = getattr(inner, 'pooling_mode', 'unknown')
    print(f"[v3 Setup] Model configured: {len(zones)} zones, "
          f"coupling={coupling_mode}, pooling={pooling_mode}")

    return zones, B_prior_cpu


def verify_v3_setup(model):
    """Check whether v3 model is properly configured.

    Call after setup_v3_for_fold() to catch configuration errors early.

    Args:
        model: TissueAwareBNT_v3 wrapper or raw TABNT_v3 instance

    Returns:
        True if setup is complete, False with warnings otherwise
    """
    inner = model.model if hasattr(model, 'model') else model
    issues = []

    # Check zones are set
    if inner._zones is None:
        issues.append("Zones not set — call setup_v3_for_fold()")

    # Check circuit pooling/readout zones
    pool = getattr(inner, 'pool', None) or getattr(inner, 'readout', None)
    if pool is not None and hasattr(pool, '_zones'):
        if pool._zones is None:
            issues.append("Circuit readout zones not set")

    # Check functional prior (handles both v3 and final model structures)
    # v3: inner.coupling.bipartite._prior_set
    # final: inner.bipartite._prior_set
    bipartite = None
    if hasattr(inner, 'bipartite'):
        bipartite = inner.bipartite
    elif hasattr(inner, 'coupling') and inner.coupling is not None:
        bipartite = getattr(inner.coupling, 'bipartite', None)

    if bipartite is not None and hasattr(bipartite, '_prior_set'):
        if not bipartite._prior_set:
            issues.append("Functional prior not set for bipartite projection")

    if issues:
        for issue in issues:
            print(f"WARNING [v3]: {issue}")
        return False

    return True