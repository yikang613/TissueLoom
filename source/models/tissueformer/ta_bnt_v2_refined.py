"""
Tissue-Aware Brain Network Transformer v2 (TA-BNT v2)

Extends BNT with:
1. Tissue-type embedding (GM vs WM node identity)
2. Bipartite WM-GM projection variants (Xu et al. 2024 inspired)

All coupling modules are inserted BETWEEN Layer 0 (no pooling) and
Layer 1 (DEC pooling 248->100), where GM/WM node identity is preserved.

Pipeline:
    Input [B, 248, 248]
        -> (optional) tissue embedding
        -> Layer 0: self-attention, no pooling [B, 248, D]
        -> (optional) bipartite projection    [B, 248, D]  <-- coupling here
        -> Layer 1: DEC pooling [B, 100, D]                <-- GM/WM split lost
        -> dim_reduction [B, 100, 8]
        -> fc [B, 800] -> [B, 2]

Ablation configs (change coupling_mode and use_tissue_embed):
    Exp 0: coupling_mode=none,               use_tissue_embed=false  -> Vanilla BNT
    Exp 1: coupling_mode=none,               use_tissue_embed=true   -> Tissue embed only
    Exp 2: coupling_mode=bipartite_learned,   use_tissue_embed=true   -> Learned coupling
    Exp 3: coupling_mode=bipartite_raw,       use_tissue_embed=true   -> Raw FC coupling
    Exp 4: coupling_mode=bipartite_combined,  use_tissue_embed=true   -> Raw + learned
    Exp 5: coupling_mode=bipartite_multihop,  use_tissue_embed=true   -> Multi-hop
    Exp 6: coupling_mode=bipartite_network,   use_tissue_embed=true   -> Network-aware
    Exp 7: coupling_mode=bipartite_raw,       use_tissue_embed=false  -> Raw only
    Exp 8: coupling_mode=bipartite_network,   use_tissue_embed=false  -> Network only
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
import numpy as np

# ----------------------------------------------------------------
# UPDATE THESE IMPORTS to match your project structure, e.g.:
from ..BNT.bnt import BrainNetworkTransformer, TransPoolingEncoder
from ..base import BaseModel
# ----------------------------------------------------------------
from ..BNT.bnt import TransPoolingEncoder
from ..base import BaseModel


# ============================================================
# Atlas Definitions
# ============================================================

NETWORK_MAPPING = {
    'Vis':         list(range(0, 14))  + list(range(100, 115)),
    'SomMot':      list(range(14, 30)) + list(range(115, 134)),
    'DorsAttn':    list(range(30, 43)) + list(range(134, 147)),
    'SalVentAttn': list(range(43, 54)) + list(range(147, 158)),
    'Limbic':      list(range(54, 60)) + list(range(158, 164)),
    'Cont':        list(range(60, 73)) + list(range(164, 182)),
    'Default':     list(range(73, 100))+ list(range(182, 200)),
}

WM_TRACT_LABELS = {
    0: 'CST_L', 1: 'ML_L', 2: 'ICBP_L', 3: 'MCBP_L', 4: 'SCBP_L',
    5: 'CP_L', 6: 'ALIC_L', 7: 'PLIC_L', 8: 'RLIC_L',
    9: 'ACR_L', 10: 'SCR_L', 11: 'PCR_L', 12: 'PTR_L',
    13: 'SS_L', 14: 'EC_L', 15: 'CGC_L', 16: 'CGH_L',
    17: 'FXC_L', 18: 'SLF_L', 19: 'SFO_L', 20: 'UF_L', 21: 'TAP_L',
    22: 'GCC', 23: 'BCC', 24: 'SCC', 25: 'FX',
    26: 'TAP_R', 27: 'UF_R', 28: 'SFO_R', 29: 'SLF_R',
    30: 'FXC_R', 31: 'CGH_R', 32: 'CGC_R', 33: 'EC_R',
    34: 'SS_R', 35: 'PTR_R', 36: 'PCR_R', 37: 'SCR_R', 38: 'ACR_R',
    39: 'RLIC_R', 40: 'PLIC_R', 41: 'ALIC_R', 42: 'CP_R',
    43: 'SCBP_R', 44: 'MCBP_R', 45: 'ICBP_R', 46: 'ML_R', 47: 'CST_R',
}

WM_TRACT_MAPPING = {
    'CST':  [0, 47],
    'ML':   [1, 46],
    'CBP':  [2, 3, 4, 43, 44, 45],
    'CP':   [5, 42],
    'ALIC': [6, 41],
    'PLIC': [7, 40],
    'RLIC': [8, 39],
    'ACR':  [9, 38],
    'SCR':  [10, 37],
    'PCR':  [11, 36],
    'PTR':  [12, 35],
    'SS':   [13, 34],
    'EC':   [14, 33],
    'CGC':  [15, 32],
    'CGH':  [16, 31],
    'FXC':  [17, 30],
    'SLF':  [18, 29],
    'SFO':  [19, 28],
    'UF':   [20, 27],
    'TAP':  [21, 26],
    'CC':   [22, 23, 24],
    'FX':   [25],
}

NETWORK_NAMES = list(NETWORK_MAPPING.keys())
TRACT_NAMES = list(WM_TRACT_MAPPING.keys())


def build_gm_network_ids(num_gm=200):
    """Build tensor of Yeo 7-network IDs for each GM parcel. Returns LongTensor [num_gm]."""
    ids = torch.zeros(num_gm, dtype=torch.long)
    for net_idx, (_, parcel_indices) in enumerate(NETWORK_MAPPING.items()):
        for p in parcel_indices:
            if p < num_gm:
                ids[p] = net_idx
    return ids


def build_wm_tract_group_ids(num_wm=48):
    """Build tensor of tract group IDs for each WM bundle. Returns LongTensor [num_wm]."""
    ids = torch.zeros(num_wm, dtype=torch.long)
    for group_idx, (_, bundle_indices) in enumerate(WM_TRACT_MAPPING.items()):
        for b in bundle_indices:
            if b < num_wm:
                ids[b] = group_idx
    return ids


# ============================================================
# Bipartite Projection Modules
# ============================================================

class BipartiteProjectionLearned(nn.Module):
    """
    Bipartite projection using learned embeddings from transformer.
    Coupling: Z_GM @ Z_WM^T in learned embedding space.
    Projection: A = B @ B^T gives WM-mediated GM-GM connectivity.

    This is the version that achieved AUC 0.611 in initial experiments.
    Extra params: 1 (alpha)
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        biadj = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)
        biadj = F.relu(biadj)

        A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_proj = A_proj * (1 - eye)

        deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A_proj / deg
        z_gm_proj = torch.matmul(A_norm, z_gm)

        z_gm_enriched = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
        return torch.cat([z_gm_enriched, z_wm], dim=1)


class BipartiteProjectionRaw(nn.Module):
    """
    Bipartite projection using raw FC values as biadjacency matrix.

    The raw input adjacency rows 0-199, columns 200-247 contain the
    actual measured WM-GM functional connectivity — exactly what
    Xu et al. (2024) showed is disrupted in preclinical AD.

    Extra params: 1 (alpha)
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]

        if node_feature_raw is not None:
            biadj = node_feature_raw[:, :self.num_gm, self.num_gm:]
        else:
            D = z_gm.shape[-1]
            biadj = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)

        biadj = F.relu(biadj)

        A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_proj = A_proj * (1 - eye)

        deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A_proj / deg
        z_gm_proj = torch.matmul(A_norm, z_gm)

        z_gm_enriched = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
        return torch.cat([z_gm_enriched, z_wm], dim=1)


class BipartiteProjectionCombined(nn.Module):
     """
     Extra params: 2 (alpha, beta) without prior,
                   3 (alpha, beta, gamma) with prior
     """
     def __init__(self, num_gm, num_wm, use_prior=False):
         super().__init__()
         self.num_gm = num_gm
         self.num_wm = num_wm
         self.alpha = nn.Parameter(torch.tensor(0.1))
         self.beta = nn.Parameter(torch.tensor(0.0))

         # Soft functional prior (Experiment B)
         self.use_prior = use_prior
         if use_prior:
             self.gamma = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1)≈0.27
         self.register_buffer('B_prior', None)
         self._prior_set = False

     def set_functional_prior(self, B_prior):
         """Set CN population prior. Called once per CV fold."""
         if isinstance(B_prior, np.ndarray):
             B_prior = torch.tensor(B_prior, dtype=torch.float)
         B_prior = B_prior / (B_prior.max() + 1e-8)
         self.B_prior = B_prior
         self._prior_set = True

     def forward(self, node_features, node_feature_raw=None):
         z_gm = node_features[:, :self.num_gm, :]
         z_wm = node_features[:, self.num_gm:, :]
         D = z_gm.shape[-1]

         biadj_learned = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)

         if node_feature_raw is not None:
             biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
             mix = torch.sigmoid(self.beta)
             biadj = mix * biadj_raw + (1 - mix) * biadj_learned
         else:
             biadj = biadj_learned

         biadj = F.relu(biadj)

         # --- Soft functional prior (Experiment B) ---
         if self.use_prior and self._prior_set and self.B_prior is not None:
             biadj = biadj + torch.sigmoid(self.gamma) * self.B_prior.unsqueeze(0)

         # --- Cache for loss (Experiment A & B) ---
         self._last_biadj = biadj

         A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
         eye = torch.eye(self.num_gm, device=z_gm.device)
         A_proj = A_proj * (1 - eye)

         deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
         A_norm = A_proj / deg
         z_gm_proj = torch.matmul(A_norm, z_gm)

         z_gm_enriched = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
         return torch.cat([z_gm_enriched, z_wm], dim=1)

     def get_mixing_weight(self):
         return torch.sigmoid(self.beta).item()

     def get_prior_weight(self):
         if self.use_prior:
             return torch.sigmoid(self.gamma).item()
         return 0.0


class BipartiteProjectionMultiHop(nn.Module):
    """
    Multi-hop WM-mediated GM connectivity.

    Hop 1: A = B @ B^T     (direct WM-mediated GM-GM connections)
    Hop 2: A^2              (two-relay WM-mediated connections)

    Weighted combination with learned per-hop importance via softmax.

    Extra params: num_hops + 1 (hop_weights + alpha)
    """
    def __init__(self, num_gm, num_wm, num_hops=2, use_raw=True):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.num_hops = num_hops
        self.use_raw = use_raw
        self.hop_weights = nn.Parameter(torch.zeros(num_hops))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        if self.use_raw and node_feature_raw is not None:
            biadj = node_feature_raw[:, :self.num_gm, self.num_gm:]
        else:
            biadj = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)

        biadj = F.relu(biadj)

        A = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A = A * (1 - eye)
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg

        weights = F.softmax(self.hop_weights, dim=0)
        z_agg = weights[0] * torch.matmul(A_norm, z_gm)

        A_power = A_norm
        for h in range(1, self.num_hops):
            A_power = torch.matmul(A_power, A_norm)
            z_agg = z_agg + weights[h] * torch.matmul(A_power, z_gm)

        z_gm_enriched = z_gm + torch.sigmoid(self.alpha) * z_agg
        return torch.cat([z_gm_enriched, z_wm], dim=1)

    def get_hop_weights(self):
        """Return learned per-hop importance."""
        return F.softmax(self.hop_weights, dim=0).detach().cpu().numpy()


class BipartiteProjectionNetwork(nn.Module):
    """
    Network-aware bipartite projection.

    Computes separate WM-mediated projections for each of the 7 Yeo
    functional networks, with learned per-network importance weights.

    Motivated by Xu et al. (2024): somatomotor, control, and dorsal
    attention networks show the most WM-GM disruption in preclinical AD.

    Extra params: num_networks + 1 = 8 (network_weights + alpha)
    """
    def __init__(self, num_gm, num_wm, use_raw=True):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.use_raw = use_raw

        network_ids = build_gm_network_ids(num_gm)
        self.register_buffer('network_ids', network_ids)
        self.num_networks = len(NETWORK_MAPPING)

        self.network_weights = nn.Parameter(torch.zeros(self.num_networks))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        B, N_gm, D = z_gm.shape

        if self.use_raw and node_feature_raw is not None:
            biadj = node_feature_raw[:, :self.num_gm, self.num_gm:]
        else:
            biadj = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)

        biadj = F.relu(biadj)

        weights = F.softmax(self.network_weights, dim=0)
        z_gm_proj = torch.zeros_like(z_gm)

        for net_id in range(self.num_networks):
            mask = (self.network_ids == net_id)
            n_nodes = mask.sum().item()
            if n_nodes < 2:
                continue

            biadj_net = biadj[:, mask, :]
            z_gm_net = z_gm[:, mask, :]

            A_net = torch.matmul(biadj_net, biadj_net.transpose(-2, -1))
            eye_net = torch.eye(n_nodes, device=z_gm.device)
            A_net = A_net * (1 - eye_net)

            deg = A_net.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            A_norm = A_net / deg

            z_net_proj = torch.matmul(A_norm, z_gm_net)
            z_gm_proj[:, mask, :] = weights[net_id] * z_net_proj

        z_gm_enriched = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
        return torch.cat([z_gm_enriched, z_wm], dim=1)

    def get_network_weights(self):
        """Return per-network importance for interpretability."""
        weights = F.softmax(self.network_weights, dim=0).detach().cpu().numpy()
        return {name: float(w) for name, w in zip(NETWORK_NAMES, weights)}

# v2.1: Add WM reverse update to BipartiteProjectionCombined
# Everything else stays EXACTLY the same as v2 Exp 4

class BipartiteProjectionCombinedBidirectional(nn.Module):
    """
    v2's combined bipartite projection + reverse WM update.
    
    Same as BipartiteProjectionCombined but also updates WM:
        GM_new = GM + sigmoid(alpha_gm) * (B @ B^T normalized) @ GM
        WM_new = WM + sigmoid(alpha_wm) * normalize(B^T) @ GM
    
    The DEC pooling in Layer 1 then sees BOTH enriched GM AND enriched WM,
    allowing clusters to capture cross-tissue patterns.
    
    Extra params: 3 (alpha_gm, alpha_wm, beta)  — only 1 more than v2 combined
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.alpha_gm = nn.Parameter(torch.tensor(0.1))
        self.alpha_wm = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        # Combined biadjacency (same as v2)
        biadj_learned = torch.matmul(
            z_gm, z_wm.transpose(-2, -1)
        ) / math.sqrt(D)

        if node_feature_raw is not None:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
            mix = torch.sigmoid(self.beta)
            biadj = mix * biadj_raw + (1 - mix) * biadj_learned
        else:
            biadj = biadj_learned

        biadj = F.relu(biadj)

        # Forward: GM ← WM-mediated GM (same as v2)
        A_gm = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_gm = A_gm * (1 - eye)
        deg = A_gm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_gm_proj = torch.matmul(A_gm / deg, z_gm)
        z_gm_new = z_gm + torch.sigmoid(self.alpha_gm) * z_gm_proj

        # Reverse: WM ← GM aggregation (NEW)
        biadj_t = biadj.transpose(-2, -1)  # [B, 48, 200]
        biadj_t_norm = biadj_t / biadj_t.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        wm_from_gm = torch.matmul(biadj_t_norm, z_gm)
        z_wm_new = z_wm + torch.sigmoid(self.alpha_wm) * wm_from_gm

        return torch.cat([z_gm_new, z_wm_new], dim=1)

    def get_mixing_weight(self):
        return torch.sigmoid(self.beta).item()

    def get_alphas(self):
        return {
            'alpha_gm': torch.sigmoid(self.alpha_gm).item(),
            'alpha_wm': torch.sigmoid(self.alpha_wm).item(),
        }
    

class BipartiteProjectionCombinedTractAttn(nn.Module):
    """
    Refinement A: Combined biadjacency + learned tract attention.
    
    Before computing the projection A = B @ B^T, weight each WM tract
    column of B by a learned importance score. This focuses the projection
    on WM tracts that are discriminative for pAD vs CN.
    
    Interpretability: after training, get_tract_group_attention() reveals
    which tract groups the model considers most important. If ALIC, PLIC,
    SCBP rank highest, this validates Xu et al.'s findings.
    
    Extra params: 48 + 2 = 50 (tract_attention + alpha + beta)
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.tract_attention = nn.Parameter(torch.zeros(num_wm))  # 48

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        # Combined biadjacency
        biadj_learned = torch.matmul(
            z_gm, z_wm.transpose(-2, -1)
        ) / math.sqrt(D)

        if node_feature_raw is not None:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
            mix = torch.sigmoid(self.beta)
            biadj = mix * biadj_raw + (1 - mix) * biadj_learned
        else:
            biadj = biadj_learned

        biadj = F.relu(biadj)  # [B, 200, 48]

        # Tract attention: weight each WM tract column
        tract_w = F.softmax(self.tract_attention, dim=0)  # [48]
        biadj = biadj * tract_w.unsqueeze(0).unsqueeze(0)  # [B, 200, 48]

        # Projection
        A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_proj = A_proj * (1 - eye)
        deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_gm_proj = torch.matmul(A_proj / deg, z_gm)

        z_gm_new = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
        return torch.cat([z_gm_new, z_wm], dim=1)  # WM unchanged

    def get_mixing_weight(self):
        return torch.sigmoid(self.beta).item()

    def get_tract_attention(self):
        weights = F.softmax(self.tract_attention, dim=0).detach().cpu().numpy()
        return {WM_TRACT_LABELS[i]: float(w) for i, w in enumerate(weights)}

    def get_tract_group_attention(self):
        weights = F.softmax(self.tract_attention, dim=0).detach().cpu().numpy()
        group_w = {}
        for name, indices in WM_TRACT_MAPPING.items():
            group_w[name] = float(sum(weights[i] for i in indices))
        return dict(sorted(group_w.items(), key=lambda x: -x[1]))


class BipartiteProjectionCombinedMultiHop(nn.Module):
    """
    Refinement B: Combined biadjacency + multi-hop projection.
    
    Merges the two best v2 variants:
    - Combined biadjacency (AUC 0.648) for biologically grounded coupling
    - Multi-hop (AUC 0.623) for network-level segregation patterns
    
    Hop 1: A = B @ B^T           (direct WM-mediated GM-GM)
    Hop 2: A^2                    (indirect: GM-WM-GM-WM-GM)
    
    Extra params: 2 + 2 = 4 (alpha + beta + 2 hop_weights)
    """
    def __init__(self, num_gm, num_wm, num_hops=2):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.num_hops = num_hops
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.hop_weights = nn.Parameter(torch.zeros(num_hops))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        # Combined biadjacency
        biadj_learned = torch.matmul(
            z_gm, z_wm.transpose(-2, -1)
        ) / math.sqrt(D)

        if node_feature_raw is not None:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
            mix = torch.sigmoid(self.beta)
            biadj = mix * biadj_raw + (1 - mix) * biadj_learned
        else:
            biadj = biadj_learned

        biadj = F.relu(biadj)

        # First-order projection
        A = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A = A * (1 - eye)
        deg = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A / deg

        # Multi-hop aggregation
        hop_w = F.softmax(self.hop_weights, dim=0)
        z_agg = hop_w[0] * torch.matmul(A_norm, z_gm)

        A_power = A_norm
        for h in range(1, self.num_hops):
            A_power = torch.matmul(A_power, A_norm)
            z_agg = z_agg + hop_w[h] * torch.matmul(A_power, z_gm)

        z_gm_new = z_gm + torch.sigmoid(self.alpha) * z_agg
        return torch.cat([z_gm_new, z_wm], dim=1)

    def get_mixing_weight(self):
        return torch.sigmoid(self.beta).item()

    def get_hop_weights(self):
        return F.softmax(self.hop_weights, dim=0).detach().cpu().numpy()


class BipartiteProjectionCombinedSparse(nn.Module):
    """
    Refinement C: Combined biadjacency + sparsification.
    
    Most WM-GM connections in the biadjacency are noise. Xu et al. found
    only 166 of ~9600 pairs were significantly disrupted. Keeping only
    top-K connections per GM node focuses the projection on meaningful
    WM-mediated pathways.
    
    Extra params: 2 (alpha + beta) — sparsification is parameter-free
    """
    def __init__(self, num_gm, num_wm, top_k=10):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.top_k = top_k  # keep top-K WM connections per GM node
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]
        D = z_gm.shape[-1]

        # Combined biadjacency
        biadj_learned = torch.matmul(
            z_gm, z_wm.transpose(-2, -1)
        ) / math.sqrt(D)

        if node_feature_raw is not None:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
            mix = torch.sigmoid(self.beta)
            biadj = mix * biadj_raw + (1 - mix) * biadj_learned
        else:
            biadj = biadj_learned

        biadj = F.relu(biadj)  # [B, 200, 48]

        # Sparsify: keep only top-K WM connections per GM node
        if self.top_k < self.num_wm:
            topk_vals, topk_idx = torch.topk(biadj, self.top_k, dim=-1)
            sparse_biadj = torch.zeros_like(biadj)
            sparse_biadj.scatter_(-1, topk_idx, topk_vals)
            biadj = sparse_biadj

        # Projection
        A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_proj = A_proj * (1 - eye)
        deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_gm_proj = torch.matmul(A_proj / deg, z_gm)

        z_gm_new = z_gm + torch.sigmoid(self.alpha) * z_gm_proj
        return torch.cat([z_gm_new, z_wm], dim=1)

    def get_mixing_weight(self):
        return torch.sigmoid(self.beta).item()
    

# ============================================================
# Normative Bipartite Projection Modules (Exp 14, 15)
# ============================================================

class BipartiteProjectionDeviation(nn.Module):
    """
    Exp 14 (revised): Normative Bipartite Projection with parallel GM-GM graphs.

    Both pathways produce GM-GM adjacency matrices and aggregate GM features,
    avoiding the cross-space interference of the original dual-pathway design.

    Pathway 1 (Absolute): A_abs = B @ B^T
        "Which GM regions share WM pathways?"

    Pathway 2 (Deviation): A_dev = ΔB_z @ ΔB_z^T
        "Which GM regions share similar WM disruption patterns?"

    For CN: ΔB_z ≈ N(0,1) noise → A_dev ≈ noise → minimal contribution
    For pAD: ΔB_z has structured negatives → A_dev captures disease pattern

    Extra params: 3 (alpha1, alpha2, beta)
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm

        # Pathway 1: absolute projection (same as Exp 4)
        self.alpha1 = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # Pathway 2: deviation projection
        self.alpha2 = nn.Parameter(torch.tensor(0.1))

        # Healthy baseline statistics (set per fold, NOT learned)
        self.register_buffer('B_healthy_mean', torch.zeros(num_gm, num_wm))
        self.register_buffer('B_healthy_std', torch.ones(num_gm, num_wm))
        self._baseline_set = False

    def set_healthy_baseline(self, B_healthy_mean, B_healthy_std):
        if isinstance(B_healthy_mean, torch.Tensor):
            self.B_healthy_mean.copy_(B_healthy_mean)
        else:
            self.B_healthy_mean.copy_(
                torch.tensor(B_healthy_mean, dtype=torch.float)
            )
        if isinstance(B_healthy_std, torch.Tensor):
            self.B_healthy_std.copy_(B_healthy_std.clamp(min=1e-6))
        else:
            self.B_healthy_std.copy_(
                torch.tensor(B_healthy_std, dtype=torch.float).clamp(min=1e-6)
            )
        self._baseline_set = True

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]   # [B, 200, D]
        z_wm = node_features[:, self.num_gm:, :]    # [B, 48, D]
        D = z_gm.shape[-1]

        # ========== Pathway 1: Standard combined projection (Exp 4) ==========
        biadj_learned = torch.matmul(
            z_gm, z_wm.transpose(-2, -1)
        ) / math.sqrt(D)

        if node_feature_raw is not None:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
            mix = torch.sigmoid(self.beta)
            biadj = mix * biadj_raw + (1 - mix) * biadj_learned
        else:
            biadj = biadj_learned
        biadj = F.relu(biadj)  # [B, 200, 48]

        A_abs = torch.matmul(biadj, biadj.transpose(-2, -1))  # [B, 200, 200]
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_abs = A_abs * (1 - eye)
        deg_abs = A_abs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_gm_abs = torch.matmul(A_abs / deg_abs, z_gm)  # [B, 200, D]

        # ========== Pathway 2: Deviation GM-GM graph ==========
        if node_feature_raw is not None and self._baseline_set:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]

            # Z-scored deviation
            delta_z = (biadj_raw - self.B_healthy_mean.unsqueeze(0)) \
                      / self.B_healthy_std.unsqueeze(0)
            # [B, 200, 48] — signed, no ReLU

            # Deviation GM-GM graph: which GM regions share disruption patterns?
            A_dev = torch.matmul(delta_z, delta_z.transpose(-2, -1))  # [B, 200, 200]
            A_dev = A_dev * (1 - eye)
            # For CN: delta_z is noise → A_dev entries are small random values
            # For pAD: delta_z has structure → A_dev has meaningful patterns

            deg_dev = A_dev.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_gm_dev = torch.matmul(A_dev / deg_dev, z_gm)  # [B, 200, D]
        else:
            z_gm_dev = torch.zeros_like(z_gm)

        # ========== Combined: both pathways aggregate GM features ==========
        z_gm_new = (z_gm
                     + torch.sigmoid(self.alpha1) * z_gm_abs
                     + torch.sigmoid(self.alpha2) * z_gm_dev)

        return torch.cat([z_gm_new, z_wm], dim=1)

    def get_mixing_weight(self):
        return torch.sigmoid(self.beta).item()

    def get_alphas(self):
        return {
            'alpha1_absolute': torch.sigmoid(self.alpha1).item(),
            'alpha2_deviation': torch.sigmoid(self.alpha2).item(),
        }

    def get_subject_deviation(self, node_feature_raw):
        biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
        delta_z = (biadj_raw - self.B_healthy_mean.unsqueeze(0)) \
                  / self.B_healthy_std.unsqueeze(0)
        tract_z = delta_z.mean(dim=1)

        tract_stats = {}
        for name, indices in WM_TRACT_MAPPING.items():
            group_z = tract_z[:, indices].mean(dim=-1)
            tract_stats[name] = {
                'mean_z': group_z.mean().item(),
                'std_z': group_z.std().item(),
            }
        return dict(sorted(tract_stats.items(),
                           key=lambda x: x[1]['mean_z']))


class BipartiteProjectionDeviationOnly(nn.Module):
    """
    Exp 15 (revised): Pure deviation GM-GM graph, no combined projection.
    """
    def __init__(self, num_gm, num_wm):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.alpha = nn.Parameter(torch.tensor(0.1))

        self.register_buffer('B_healthy_mean', torch.zeros(num_gm, num_wm))
        self.register_buffer('B_healthy_std', torch.ones(num_gm, num_wm))
        self._baseline_set = False

    def set_healthy_baseline(self, B_healthy_mean, B_healthy_std):
        if isinstance(B_healthy_mean, torch.Tensor):
            self.B_healthy_mean.copy_(B_healthy_mean)
        else:
            self.B_healthy_mean.copy_(
                torch.tensor(B_healthy_mean, dtype=torch.float)
            )
        if isinstance(B_healthy_std, torch.Tensor):
            self.B_healthy_std.copy_(B_healthy_std.clamp(min=1e-6))
        else:
            self.B_healthy_std.copy_(
                torch.tensor(B_healthy_std, dtype=torch.float).clamp(min=1e-6)
            )
        self._baseline_set = True

    def forward(self, node_features, node_feature_raw=None):
        z_gm = node_features[:, :self.num_gm, :]
        z_wm = node_features[:, self.num_gm:, :]

        if node_feature_raw is not None and self._baseline_set:
            biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]

            delta_z = (biadj_raw - self.B_healthy_mean.unsqueeze(0)) \
                      / self.B_healthy_std.unsqueeze(0)

            A_dev = torch.matmul(delta_z, delta_z.transpose(-2, -1))
            eye = torch.eye(self.num_gm, device=z_gm.device)
            A_dev = A_dev * (1 - eye)

            deg_dev = A_dev.abs().sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_gm_dev = torch.matmul(A_dev / deg_dev, z_gm)
        else:
            z_gm_dev = torch.zeros_like(z_gm)

        z_gm_new = z_gm + torch.sigmoid(self.alpha) * z_gm_dev
        return torch.cat([z_gm_new, z_wm], dim=1)

    def get_subject_deviation(self, node_feature_raw):
        biadj_raw = node_feature_raw[:, :self.num_gm, self.num_gm:]
        delta_z = (biadj_raw - self.B_healthy_mean.unsqueeze(0)) \
                  / self.B_healthy_std.unsqueeze(0)
        tract_z = delta_z.mean(dim=1)

        tract_stats = {}
        for name, indices in WM_TRACT_MAPPING.items():
            group_z = tract_z[:, indices].mean(dim=-1)
            tract_stats[name] = {
                'mean_z': group_z.mean().item(),
                'std_z': group_z.std().item(),
            }
        return dict(sorted(tract_stats.items(),
                           key=lambda x: x[1]['mean_z']))



# ============================================================
# Coupling Module Factory
# ============================================================

def build_coupling_module(coupling_mode, num_gm, num_wm, bipartite_hops=2, sparse_top_k=10, use_prior=False):
    """Factory function to build the appropriate coupling module."""
    if coupling_mode == 'none':
        return None
    elif coupling_mode == 'bipartite_learned':
        return BipartiteProjectionLearned(num_gm, num_wm)
    elif coupling_mode == 'bipartite_raw':
        return BipartiteProjectionRaw(num_gm, num_wm)
    elif coupling_mode == 'bipartite_combined':
        return BipartiteProjectionCombined(num_gm, num_wm, use_prior=use_prior)
    elif coupling_mode == 'bipartite_multihop':
        return BipartiteProjectionMultiHop(num_gm, num_wm,
                                           num_hops=bipartite_hops, use_raw=True)
    
        # Add to build_coupling_module():
    elif coupling_mode == 'bipartite_combined_tract_attn':
        return BipartiteProjectionCombinedTractAttn(num_gm, num_wm)
    elif coupling_mode == 'bipartite_combined_multihop':
        return BipartiteProjectionCombinedMultiHop(num_gm, num_wm, num_hops=bipartite_hops)
    elif coupling_mode == 'bipartite_combined_sparse':
        return BipartiteProjectionCombinedSparse(num_gm, num_wm, top_k=sparse_top_k)
    
    # Add to build_coupling_module():
    elif coupling_mode == 'bipartite_combined_bidirectional':
        return BipartiteProjectionCombinedBidirectional(num_gm, num_wm)
    elif coupling_mode == 'bipartite_network':
        return BipartiteProjectionNetwork(num_gm, num_wm, use_raw=True)
    
    elif coupling_mode == 'bipartite_deviation':
        return BipartiteProjectionDeviation(num_gm, num_wm)
    elif coupling_mode == 'bipartite_deviation_only':
        return BipartiteProjectionDeviationOnly(num_gm, num_wm)

    else:
        raise ValueError(
            f"Unknown coupling_mode: {coupling_mode}. "
            f"Options: none, bipartite_learned, bipartite_raw, "
            f"bipartite_combined, bipartite_multihop, bipartite_combined_bidirectional, bipartite_network, "
            f"bipartite_deviation, bipartite_deviation_only"
        )


def resolve_ta_bnt_v2_model_cfg(model_cfg: DictConfig):
    """Resolve flat model config or experiment-selected nested config."""
    selected_experiment = None
    if hasattr(model_cfg, 'experiment'):
        selected_experiment = model_cfg.experiment
    elif hasattr(model_cfg, 'exp_name'):
        selected_experiment = model_cfg.exp_name

    if selected_experiment and hasattr(model_cfg, selected_experiment):
        return getattr(model_cfg, selected_experiment)

    return model_cfg


# ============================================================
# Main Model
# ============================================================

class TissueAwareBNT_v2(BaseModel):
    """
    Tissue-Aware Brain Network Transformer v2.

    Supports all bipartite projection variants via a single config field.

    Config example:
        model:
            name: TissueAwareBNT_v2
            sizes: [360, 100]
            pooling: [false, true]
            pos_encoding: none
            orthogonal: true
            freeze_center: true
            project_assignment: true
            pos_embed_dim: 360
            num_gm_nodes: 200
            num_wm_nodes: 48
            coupling_mode: bipartite_raw
            use_tissue_embed: true
            bipartite_hops: 2            # only needed for multihop
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        model_cfg = resolve_ta_bnt_v2_model_cfg(config.model)

        # --- Config ---
        self.num_gm_nodes = getattr(model_cfg, 'num_gm_nodes', 200)
        self.num_wm_nodes = getattr(model_cfg, 'num_wm_nodes', 48)
        self.coupling_mode = getattr(model_cfg, 'coupling_mode', 'none')
        self.use_tissue_embed = getattr(model_cfg, 'use_tissue_embed', True)
        self.pos_encoding = model_cfg.pos_encoding
        bipartite_hops = getattr(model_cfg, 'bipartite_hops', 2)
        sparse_top_k = getattr(model_cfg, 'sparse_top_k', 10)
        use_prior = getattr(model_cfg, 'use_prior', False)

        # --- Feature dimension ---
        forward_dim = config.dataset.node_sz  # 248

        # --- Positional encoding (identity mode) ---
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, model_cfg.pos_embed_dim),
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + model_cfg.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # --- Tissue embedding ---
        if self.use_tissue_embed:
            if self.pos_encoding == 'identity':
                tissue_dim = model_cfg.pos_embed_dim
            else:
                tissue_dim = config.dataset.node_sz  # 248
            self.tissue_embed = nn.Embedding(2, tissue_dim)
            nn.init.zeros_(self.tissue_embed.weight)

        # --- BNT encoder layers ---
        sizes = model_cfg.sizes.copy()
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = model_cfg.pooling
        self.do_pooling = do_pooling

        self.attention_list = nn.ModuleList()
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[index],
                    hidden_size=1024,
                    output_node_num=size,
                    pooling=do_pooling[index],
                    orthogonal=model_cfg.orthogonal,
                    freeze_center=model_cfg.freeze_center,
                    project_assignment=model_cfg.project_assignment
                )
            )

        # --- Coupling module ---
        self.coupling_module = build_coupling_module(
            self.coupling_mode,
            self.num_gm_nodes,
            self.num_wm_nodes,
            bipartite_hops,
            sparse_top_k,
            use_prior=use_prior
        )

        # --- Standard BNT readout ---
        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(self, time_seires, node_feature):
        """
        Args:
            time_seires: [B, T, N] raw time series (kept for API compatibility)
            node_feature: [B, 248, 248] functional connectivity matrix
        Returns:
            logits: [B, 2]
            assignments: list of assignment matrices from pooling layers
        """
        bz = node_feature.shape[0]
        device = node_feature.device

        # Save raw input for bipartite modules that need raw FC
        node_feature_raw = node_feature  # [B, 248, 248]

        # === Encoding ===
        if self.pos_encoding == 'identity':
            if self.use_tissue_embed:
                tissue_ids = torch.cat([
                    torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
                    torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
                ])
                tissue_emb = self.tissue_embed(tissue_ids)
                combined_identity = self.node_identity + tissue_emb
                pos_emb = combined_identity.unsqueeze(0).expand(bz, -1, -1)
            else:
                pos_emb = self.node_identity.unsqueeze(0).expand(bz, -1, -1)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        elif self.pos_encoding == 'none':
            if self.use_tissue_embed:
                tissue_ids = torch.cat([
                    torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
                    torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
                ])
                tissue_emb = self.tissue_embed(tissue_ids)
                node_feature = node_feature + tissue_emb.unsqueeze(0)

        # === Layer 0: Self-attention, NO pooling (248 nodes preserved) ===
        node_feature, assignment_0 = self.attention_list[0](node_feature)

        # === Bipartite projection BEFORE pooling ===
        if self.coupling_module is not None:
            node_feature = self.coupling_module(
                node_feature, node_feature_raw=node_feature_raw
            )

        # === Layer 1: DEC pooling 248 -> 100 ===
        node_feature, assignment_1 = self.attention_list[1](node_feature)

        # === Standard readout ===
        node_feature = self.dim_reduction(node_feature)
        combined = node_feature.reshape(bz, -1)

        return self.fc(combined), [assignment_0, assignment_1]

    # ============================================================
    # Standard BNT interface
    # ============================================================

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_biadj(self):
       if self.coupling_module is not None:
           return getattr(self.coupling_module, '_last_biadj', None)
       return None

    def loss(self, assignments):
        """Compute auxiliary loss (KL divergence for DEC clustering)."""
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None
        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all

    def get_cluster_centers(self):
        """Get DEC cluster centers from pooling layers."""
        centers = []
        for atten in self.attention_list:
            if atten.is_pooling_enabled():
                centers.append(atten.dec.get_cluster_centers())
        return centers

    def get_biadj(self):
        """Retrieve cached biadjacency matrix from last forward pass."""
        if self.coupling_module is not None:
            return getattr(self.coupling_module, '_last_biadj', None)
        return None

    # ============================================================
    # Interpretability
    # ============================================================

    def get_tissue_embedding(self):
        """Get learned tissue embeddings for analysis."""
        if not self.use_tissue_embed:
            return None
        gm_emb = self.tissue_embed.weight[0].detach().cpu()
        wm_emb = self.tissue_embed.weight[1].detach().cpu()
        diff = wm_emb - gm_emb
        return {
            'GM': gm_emb.numpy(),
            'WM': wm_emb.numpy(),
            'difference': diff.numpy(),
            'GM_norm': gm_emb.norm().item(),
            'WM_norm': wm_emb.norm().item(),
            'diff_norm': diff.norm().item(),
            'cosine_similarity': F.cosine_similarity(
                gm_emb.unsqueeze(0), wm_emb.unsqueeze(0)
            ).item()
        }

    def get_bipartite_info(self):
        """Get coupling module learned parameters for analysis."""
        info = {
            'coupling_mode': self.coupling_mode,
            'use_tissue_embed': self.use_tissue_embed,
        }

        if self.coupling_module is None:
            return info

        if hasattr(self.coupling_module, 'alpha'):
            info['alpha'] = torch.sigmoid(self.coupling_module.alpha).item()

        if hasattr(self.coupling_module, 'get_mixing_weight'):
            info['raw_vs_learned_mix'] = self.coupling_module.get_mixing_weight()

        if hasattr(self.coupling_module, 'get_hop_weights'):
            hop_w = self.coupling_module.get_hop_weights()
            info['hop_weights'] = {
                f'hop_{i+1}': float(w) for i, w in enumerate(hop_w)
            }

        if hasattr(self.coupling_module, 'get_network_weights'):
            info['network_weights'] = self.coupling_module.get_network_weights()

        return info

    def get_coupling_matrix(self, time_seires, node_feature):
        """
        Extract learned and raw WM-GM coupling matrices for a batch.

        Returns dict with:
            'learned_coupling': np.array [B, 200, 48]
            'raw_coupling': np.array [B, 200, 48]
        """
        self.eval()
        with torch.no_grad():
            device = node_feature.device
            bz = node_feature.shape[0]
            node_feature_raw = node_feature

            if self.pos_encoding == 'identity':
                if self.use_tissue_embed:
                    tissue_ids = torch.cat([
                        torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
                        torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
                    ])
                    tissue_emb = self.tissue_embed(tissue_ids)
                    combined_identity = self.node_identity + tissue_emb
                    pos_emb = combined_identity.unsqueeze(0).expand(bz, -1, -1)
                else:
                    pos_emb = self.node_identity.unsqueeze(0).expand(bz, -1, -1)
                node_feature = torch.cat([node_feature, pos_emb], dim=-1)
            elif self.pos_encoding == 'none' and self.use_tissue_embed:
                tissue_ids = torch.cat([
                    torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
                    torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
                ])
                tissue_emb = self.tissue_embed(tissue_ids)
                node_feature = node_feature + tissue_emb.unsqueeze(0)

            node_feature, _ = self.attention_list[0](node_feature)

            z_gm = node_feature[:, :self.num_gm_nodes, :]
            z_wm = node_feature[:, self.num_gm_nodes:, :]
            D = z_gm.shape[-1]
            learned_coupling = torch.matmul(
                z_gm, z_wm.transpose(-2, -1)
            ) / math.sqrt(D)

            raw_coupling = node_feature_raw[:, :self.num_gm_nodes, self.num_gm_nodes:]

        return {
            'learned_coupling': learned_coupling.cpu().numpy(),
            'raw_coupling': raw_coupling.cpu().numpy(),
        }

    def count_extra_params(self):
        """Count parameters added beyond vanilla BNT."""
        extra = 0
        if self.use_tissue_embed:
            extra += sum(p.numel() for p in self.tissue_embed.parameters())
        if self.coupling_module is not None:
            extra += sum(p.numel() for p in self.coupling_module.parameters())
        return extra

    def get_config_summary(self):
        """Return a summary dict of model configuration."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'model': 'TissueAwareBNT_v2',
            'pos_encoding': self.pos_encoding,
            'use_tissue_embed': self.use_tissue_embed,
            'coupling_mode': self.coupling_mode,
            'num_gm_nodes': self.num_gm_nodes,
            'num_wm_nodes': self.num_wm_nodes,
            'extra_params': self.count_extra_params(),
            'total_params': total,
            'trainable_params': trainable,
        }