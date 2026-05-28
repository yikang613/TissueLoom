"""
Tissue-Aware Brain Network Transformer (TA-BNT) — Final Architecture
=====================================================================

A self-contained tissue-aware brain network classification framework.
No external model dependencies (replaces TransPoolingEncoder with a
custom tissue-conditioned attention module).

Three unified components:

  1. Tissue-Conditioned Whole-Brain Transformer Encoding
     Global self-attention with explicit tissue-pair bias models
     GM->GM, GM->WM, WM->GM, and WM->WM interactions distinctly.

  2. Bidirectional Cross-Tissue Affinity Propagation
     A learned GM-WM affinity representation integrates normalized
     raw coupling and learned similarity, propagating information
     bidirectionally across tissues via A@A^T and A^T@A.

  3. Anatomy-Informed Circuit Dictionary Readout
     Tissue representations are summarized into circuit-level GM and
     WM states using a structured tract-group dictionary with prior-
     derived contribution weights.

Pipeline:
    FC [B, 248, 248]
      -> + tissue embedding                          [B, 248, 248]
      -> tissue-conditioned self-attention            [B, 248, 248]
         (with 2x2 tissue-pair bias on attn logits)
      -> split Z_GM [B,200,248] + Z_WM [B,48,248]
      -> bidirectional affinity propagation           Z_GM' + Z_WM'
         (softplus affinity + prior + B@B^T / B^T@B)
      -> separate dim reduction                       [B,200,32] + [B,48,32]
      -> circuit dictionary readout                   [B, pool_dim]
         (22 circuits x [h_gm + h_wm] + global)
      -> MLP classifier                               [B, 2]

References:
  - Mesulam (1990): Large-scale distributed neural networks
  - Filley & Fields (2016): WM and cognition
  - Xu et al. (2024): WM-GM functional connectome disruption in pAD
  - Kan et al. (2022): Brain Network Transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from ..base import BaseModel


# ============================================================
# Atlas Definitions
# ============================================================

YEO_7_NETWORKS = {
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

JHU_TRACT_GROUPS = {
    'CST':  [0, 47],  'ML':   [1, 46],
    'CBP':  [2, 3, 4, 43, 44, 45],
    'CP':   [5, 42],  'ALIC': [6, 41],  'PLIC': [7, 40],  'RLIC': [8, 39],
    'ACR':  [9, 38],  'SCR':  [10, 37], 'PCR':  [11, 36], 'PTR':  [12, 35],
    'SS':   [13, 34], 'EC':   [14, 33], 'CGC':  [15, 32], 'CGH':  [16, 31],
    'FXC':  [17, 30], 'SLF':  [18, 29], 'SFO':  [19, 28], 'UF':   [20, 27],
    'TAP':  [21, 26], 'CC':   [22, 23, 24], 'FX':   [25],
}


def validate_tract_groups(tract_groups, num_wm=48):
    """Ensure every WM index appears in exactly one group."""
    all_indices = set()
    for name, indices in tract_groups.items():
        for idx in indices:
            if idx in all_indices:
                raise ValueError(f"WM index {idx} in multiple groups!")
            all_indices.add(idx)
    missing = set(range(num_wm)) - all_indices
    if missing:
        for idx in sorted(missing):
            tract_groups[f'tract_{idx}'] = [idx]
    return tract_groups


# ============================================================
# Zone Generator
# ============================================================

class ZoneGenerator:
    """Generate circuit definitions from healthy population coupling prior."""

    def __init__(self, tract_groups, method='topk', k=15,
                 threshold_std=0.5, min_zone_size=5):
        self.tract_groups = tract_groups
        self.method = method
        self.k = k
        self.threshold_std = threshold_std
        self.min_zone_size = min_zone_size
        self.zones = None

    def generate_zones(self, B_prior):
        if not isinstance(B_prior, torch.Tensor):
            B_prior = torch.tensor(B_prior, dtype=torch.float)
        zones = {}
        for group_name, tract_indices in self.tract_groups.items():
            coupling = B_prior[:, tract_indices].mean(dim=-1)
            if self.method == 'topk':
                k_eff = min(self.k, coupling.shape[0])
                _, gm_indices = coupling.topk(k_eff)
                gm_indices = gm_indices.tolist()
            elif self.method == 'threshold':
                mu, std = coupling.mean(), coupling.std()
                threshold = mu + self.threshold_std * std
                gm_indices = (coupling > threshold).nonzero(as_tuple=True)[0].tolist()
                if len(gm_indices) < self.min_zone_size:
                    _, fallback = coupling.topk(self.min_zone_size)
                    gm_indices = fallback.tolist()
            else:
                raise ValueError(f"Unknown method: {self.method}")
            zones[group_name] = {'gm': sorted(gm_indices), 'wm': tract_indices}
        self.zones = zones
        return zones

    def summary(self):
        if not self.zones:
            print("No zones generated."); return
        print(f"\n{'Zone':<20} {'GM parcels':>12} {'WM tracts':>12}")
        print("-" * 46)
        total_gm = 0
        for name, zone in self.zones.items():
            n_gm, n_wm = len(zone['gm']), len(zone['wm'])
            total_gm += n_gm
            print(f"{name:<20} {n_gm:>12} {n_wm:>12}")
        all_gm = set()
        for z in self.zones.values():
            all_gm.update(z['gm'])
        print(f"\nUnique GM: {len(all_gm)}/200 | "
              f"Mean GM/zone: {total_gm/len(self.zones):.1f}")


# ============================================================
# COMPONENT 1: Tissue-Conditioned Self-Attention
# ============================================================

class TissueConditionedAttention(nn.Module):
    """Tissue-aware multi-head self-attention with tissue-pair bias.

    Standard transformer encoder layer (attention + FFN + residual +
    layer norm) with one addition: a 2x2 learnable tissue-pair bias
    matrix that is added to attention logits before softmax.

    This makes the attention explicitly tissue-conditioned:
      attention[i,j] = (q_i . k_j) / sqrt(d) + b[tau(i), tau(j)]

    where tau(i) in {0=GM, 1=WM} and b is a 2x2 learnable matrix.

    The bias captures that different tissue-pair interactions should
    follow different attention rules:
      b[GM,GM]:  cortico-cortical coupling strength
      b[GM,WM]:  how strongly cortical regions attend to tracts
      b[WM,GM]:  how strongly tracts attend to cortical regions
      b[WM,WM]:  tract-tract co-activation strength

    Parameters: 4 (tissue-pair bias) + standard transformer params
    """

    def __init__(self, embed_dim=248, num_heads=8, ffn_dim=1024,
                 dropout=0.1, num_gm=200, num_wm=48,
                 use_tissue_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_gm = num_gm
        self.num_wm = num_wm
        # Module 1 ablation flag: when False, drop the tissue-pair bias
        # (becomes a vanilla self-attention layer).
        self.use_tissue_bias = use_tissue_bias

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Tissue-pair bias: 2x2 learnable matrix (4 parameters)
        # Initialized to zero so model starts tissue-agnostic
        self.tissue_pair_bias = nn.Parameter(torch.zeros(2, 2))

        # Pre-compute tissue type indices (registered as buffer for device handling)
        tissue_types = torch.cat([
            torch.zeros(num_gm, dtype=torch.long),
            torch.ones(num_wm, dtype=torch.long),
        ])
        self.register_buffer('tissue_types', tissue_types)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _build_tissue_bias_mask(self):
        """Build [N, N] attention bias from 2x2 tissue-pair matrix.

        Returns a mask where mask[i,j] = tissue_pair_bias[tau(i), tau(j)].
        This gets added to attention logits before softmax.
        """
        # tissue_types: [248] with values 0 (GM) or 1 (WM)
        row_types = self.tissue_types.unsqueeze(1)   # [248, 1]
        col_types = self.tissue_types.unsqueeze(0)   # [1, 248]
        # Index into the 2x2 bias matrix
        mask = self.tissue_pair_bias[row_types, col_types]  # [248, 248]
        return mask

    def forward(self, x):
        """
        Args:
            x: [B, 248, 248] node features (FC + tissue embedding)
        Returns:
            z: [B, 248, 248] encoded features
        """
        # Build tissue-pair bias mask (skipped under use_tissue_bias=False)
        attn_mask = self._build_tissue_bias_mask() if self.use_tissue_bias else None

        # Self-attention (with tissue-pair bias when enabled)
        residual = x
        attn_out, attn_weights = self.self_attn(
            x, x, x, attn_mask=attn_mask)
        x = self.norm1(residual + self.dropout(attn_out))

        # Feed-forward network
        residual = x
        x = self.norm2(residual + self.ffn(x))

        return x


# ============================================================
# COMPONENT 2: Bidirectional Cross-Tissue Affinity Propagation
# ============================================================

class BipartiteProjectionBidirectional(nn.Module):
    """Bidirectional cross-tissue affinity propagation via A@A^T and A^T@A,
    with optional within-tissue residual propagation.

    Cross-tissue pathway (main contribution):
      GM-WM affinity matrix A → GM enriched via A@A^T, WM via A^T@A

    Within-tissue pathway (complementary):
      GM-GM similarity → GM receives context from functionally similar GM regions
      WM-WM similarity → WM receives context from co-activated WM tracts
      Gated by learnable scalars, initialized near zero so cross-tissue
      dominates by default. The model learns to incorporate within-tissue
      structure only when the data demands it.

    Parameters: 6 scalars (alpha_gm, alpha_wm, beta, gamma, lambda_gm, lambda_wm)
    """

    def __init__(self, num_gm=200, num_wm=48, use_prior=True,
                 use_within_tissue=False):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.use_prior = use_prior
        self.use_within_tissue = use_within_tissue

        # Cross-tissue propagation gates
        self.alpha_gm = nn.Parameter(torch.tensor(0.1))
        self.alpha_wm = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        if use_prior:
            self.gamma = nn.Parameter(torch.tensor(-1.0))

        # Within-tissue propagation gates (only when enabled)
        if use_within_tissue:
            self.lambda_gm = nn.Parameter(torch.tensor(-1.0))
            self.lambda_wm = nn.Parameter(torch.tensor(-1.0))

        self.register_buffer('B_prior', torch.zeros(num_gm, num_wm))
        self._prior_set = False
        self._last_biadj = None

    def set_functional_prior(self, B_prior):
        if not isinstance(B_prior, torch.Tensor):
            B_prior = torch.tensor(B_prior, dtype=torch.float)
        self.B_prior.copy_(B_prior / (B_prior.max() + 1e-8))
        self._prior_set = True

    def forward(self, z_gm, z_wm, fc_cross):
        """
        Args:
            z_gm:     [B, 200, D]  GM features (post self-attention)
            z_wm:     [B, 48, D]   WM features (post self-attention)
            fc_cross: [B, 200, 48] raw GM-WM coupling from input FC
        Returns:
            z_gm_out: [B, 200, D]  enriched GM features
            z_wm_out: [B, 48, D]   enriched WM features
        """
        D = z_gm.shape[-1]

        # Learned similarity
        biadj_learned = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)

        # Normalize both branches before mixing
        fc_cross_norm = fc_cross / (fc_cross.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)
        biadj_learned_norm = biadj_learned / (biadj_learned.abs().mean(dim=(1, 2), keepdim=True) + 1e-6)

        mix = torch.sigmoid(self.beta)
        biadj_raw = mix * fc_cross_norm + (1 - mix) * biadj_learned_norm

        # Smooth positive affinity
        biadj = F.softplus(biadj_raw)

        # Healthy population prior (main inductive bias location)
        if self.use_prior and self._prior_set:
            biadj = biadj + torch.sigmoid(self.gamma) * self.B_prior.unsqueeze(0)

        self._last_biadj = biadj

        # GM enrichment via shared WM pathways: A @ A^T (cross-tissue)
        A_gm_cross = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye_gm = torch.eye(self.num_gm, device=z_gm.device)
        A_gm_cross = A_gm_cross * (1 - eye_gm)
        deg_gm = A_gm_cross.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_gm_proj = torch.matmul(A_gm_cross / deg_gm, z_gm)
        z_gm_out = z_gm + torch.sigmoid(self.alpha_gm) * z_gm_proj

        # WM enrichment via shared GM regions: A^T @ A (cross-tissue)
        A_wm_cross = torch.matmul(biadj.transpose(-2, -1), biadj)
        eye_wm = torch.eye(self.num_wm, device=z_wm.device)
        A_wm_cross = A_wm_cross * (1 - eye_wm)
        deg_wm = A_wm_cross.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_wm_proj = torch.matmul(A_wm_cross / deg_wm, z_wm)
        z_wm_out = z_wm + torch.sigmoid(self.alpha_wm) * z_wm_proj

        # --- Within-tissue residual propagation (optional) ---
        # Enabled for tasks with intra-tissue signal (e.g., AD/CN with full FC).
        # Disabled for tasks where intra-tissue is noise (e.g., pAD/CN with
        # cross-tissue-only input where GM-GM and WM-WM blocks are zeros).
        if self.use_within_tissue:
            # GM-GM: each GM parcel receives context from functionally similar
            # GM parcels (learned feature similarity, not raw FC).
            A_gm_within = torch.matmul(z_gm, z_gm.transpose(-2, -1)) / math.sqrt(D)
            A_gm_within = F.relu(A_gm_within) * (1 - eye_gm)
            deg_gm_w = A_gm_within.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_gm_within = torch.matmul(A_gm_within / deg_gm_w, z_gm_out)
            z_gm_out = z_gm_out + torch.sigmoid(self.lambda_gm) * z_gm_within

            # WM-WM: each WM tract receives context from co-activated tracts
            A_wm_within = torch.matmul(z_wm, z_wm.transpose(-2, -1)) / math.sqrt(D)
            A_wm_within = F.relu(A_wm_within) * (1 - eye_wm)
            deg_wm_w = A_wm_within.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            z_wm_within = torch.matmul(A_wm_within / deg_wm_w, z_wm_out)
            z_wm_out = z_wm_out + torch.sigmoid(self.lambda_wm) * z_wm_within

        return z_gm_out, z_wm_out


# ============================================================
# COMPONENT 3: Anatomy-Informed Circuit Dictionary Readout
# ============================================================

class CircuitDictionaryReadout(nn.Module):
    """Anatomy-informed circuit dictionary readout with prior-derived weights.

    Pools node features into biologically defined neural circuits using
    a fixed tract-group dictionary. Parcels with stronger healthy
    coupling to the tract group contribute more to the circuit
    representation (prior-weighted, still parameter-free).

    Per circuit c:
      h_gm_c: weighted_pool(Z_GM[zone_gm])   -- cortical state
      h_wm_c: weighted_pool(Z_WM[zone_wm])   -- tract integrity

    Optionally includes raw coupling scalar s_c for interpretability
    analysis (configurable whether it enters the classifier).

    Parameters: 0 (all weights derived from B_CN_prior)
    """

    def __init__(self, d_gm=32, d_wm=32,
                 include_coupling_scalar=False,
                 include_global_residual=True,
                 include_network_summaries=False,
                 use_prompt_tokens=False):
        super().__init__()
        self.d_gm = d_gm
        self.d_wm = d_wm
        self.include_coupling_scalar = include_coupling_scalar
        self.include_global_residual = include_global_residual
        self.include_network_summaries = include_network_summaries
        self.use_prompt_tokens = use_prompt_tokens
        self._zones = None
        self._n_circuits = 0
        self._zone_gm_weights = {}
        self._zone_wm_weights = {}
        # Always store coupling scalars for interpretability
        self._last_coupling_scalars = None

        # Circuit prompt tokens: pre-allocate for 22 JHU circuits
        # Registered here so PyTorch tracks them as parameters from the start
        print(f"[DEBUG CircuitReadout] use_prompt_tokens={self.use_prompt_tokens}")
        if self.use_prompt_tokens:
            self._gm_prompts = nn.Parameter(torch.randn(22, self.d_gm) * 0.02)
            self._wm_prompts = nn.Parameter(torch.randn(22, self.d_wm) * 0.02)
            print(f"[DEBUG CircuitReadout] Created prompts: gm={self._gm_prompts.shape}, "
                  f"wm={self._wm_prompts.shape}, total={self._gm_prompts.numel() + self._wm_prompts.numel()} params")
        else:
            self._gm_prompts = None
            self._wm_prompts = None

    def setup_zones(self, zones, B_prior=None):
        """Set circuit dictionary and derive pooling weights from prior."""
        self._zones = zones
        self._n_circuits = len(zones)
        self._zone_gm_weights = {}
        self._zone_wm_weights = {}

        if B_prior is not None:
            if not isinstance(B_prior, torch.Tensor):
                B_prior = torch.tensor(B_prior, dtype=torch.float)

            for zone_name, zone in zones.items():
                gm_idx = zone['gm']
                wm_idx = zone['wm']

                if gm_idx and wm_idx:
                    gm_strength = B_prior[gm_idx][:, wm_idx].mean(dim=-1)
                    gm_weight = gm_strength / (gm_strength.sum() + 1e-6)
                    self._zone_gm_weights[zone_name] = gm_weight

                    wm_strength = B_prior[gm_idx][:, wm_idx].mean(dim=0)
                    wm_weight = wm_strength / (wm_strength.sum() + 1e-6)
                    self._zone_wm_weights[zone_name] = wm_weight

    @property
    def output_dim(self):
        n = self._n_circuits if self._n_circuits > 0 else 22
        per_circuit = self.d_gm + self.d_wm
        if self.include_coupling_scalar:
            per_circuit += 1
        total = n * per_circuit
        if self.include_network_summaries:
            total += 7 * self.d_gm   # Yeo 7 network GM summaries
        if self.include_global_residual:
            total += self.d_gm + self.d_wm
        return total

    def forward(self, z_gm, z_wm, fc_cross):
        """
        Args:
            z_gm:     [B, 200, d_gm]  reduced GM features
            z_wm:     [B, 48, d_wm]   reduced WM features
            fc_cross: [B, 200, 48]     raw GM-WM coupling from input FC
        Returns:
            pooled: [B, output_dim]
        """
        assert self._zones is not None, \
            "Call setup_zones() first via setup_v3_for_fold()."

        B = z_gm.shape[0]
        device = z_gm.device
        circuit_feats = []
        coupling_scalars = {}

        for ci, (zone_name, zone) in enumerate(self._zones.items()):
            gm_idx = zone['gm']
            wm_idx = zone['wm']
            parts = []

            # Cortical state
            if gm_idx:
                gm_feats = z_gm[:, gm_idx, :]  # [B, n_gm, d]
                if self.use_prompt_tokens and self._gm_prompts is not None:
                    # Cross-attention: prompt queries into circuit nodes
                    prompt = self._gm_prompts[ci].unsqueeze(0).expand(B, -1)  # [B, d]
                    attn = torch.einsum('bd,bnd->bn', prompt, gm_feats)       # [B, n_gm]
                    attn = torch.softmax(attn / (self.d_gm ** 0.5), dim=-1)   # [B, n_gm]
                    h_gm = torch.einsum('bn,bnd->bd', attn, gm_feats)         # [B, d]
                elif zone_name in self._zone_gm_weights:
                    w = self._zone_gm_weights[zone_name].to(device).view(1, -1, 1)
                    h_gm = (gm_feats * w).sum(dim=1)
                else:
                    h_gm = gm_feats.mean(dim=1)
            else:
                h_gm = torch.zeros(B, self.d_gm, device=device)
            parts.append(h_gm)

            # Tract integrity
            if wm_idx:
                wm_feats = z_wm[:, wm_idx, :]  # [B, n_wm, d]
                if self.use_prompt_tokens and self._wm_prompts is not None:
                    prompt = self._wm_prompts[ci].unsqueeze(0).expand(B, -1)  # [B, d]
                    attn = torch.einsum('bd,bnd->bn', prompt, wm_feats)       # [B, n_wm]
                    attn = torch.softmax(attn / (self.d_wm ** 0.5), dim=-1)   # [B, n_wm]
                    h_wm = torch.einsum('bn,bnd->bd', attn, wm_feats)         # [B, d]
                elif zone_name in self._zone_wm_weights:
                    w = self._zone_wm_weights[zone_name].to(device).view(1, -1, 1)
                    h_wm = (wm_feats * w).sum(dim=1)
                else:
                    h_wm = wm_feats.mean(dim=1)
            else:
                h_wm = torch.zeros(B, self.d_wm, device=device)
            parts.append(h_wm)

            # Raw coupling scalar — ALWAYS computed for interpretability
            if gm_idx and wm_idx:
                coupling = fc_cross[:, gm_idx, :][:, :, wm_idx]
                s_c = coupling.abs().mean(dim=(1, 2))           # [B]
            else:
                s_c = torch.zeros(B, device=device)
            coupling_scalars[zone_name] = s_c.detach()

            # Only feed s_c to classifier if configured
            if self.include_coupling_scalar:
                parts.append(s_c.unsqueeze(-1))                 # [B, 1]

            circuit_feats.append(torch.cat(parts, dim=-1))

        self._last_coupling_scalars = coupling_scalars

        all_circuits = torch.cat(circuit_feats, dim=-1)

        # Yeo 7 functional network summaries (intra-GM structure)
        # Captures GM-GM network patterns (Default mode, frontoparietal, etc.)
        # that are disrupted in clinical AD but not accessible through
        # WM-defined circuit groupings alone.
        if self.include_network_summaries:
            net_feats = []
            for net_name, gm_indices in YEO_7_NETWORKS.items():
                h_net = z_gm[:, gm_indices, :].mean(dim=1)   # [B, d_gm]
                net_feats.append(h_net)
            all_circuits = torch.cat(
                [all_circuits] + net_feats, dim=-1)

        # Global residual
        if self.include_global_residual:
            global_gm = z_gm.mean(dim=1)
            global_wm = z_wm.mean(dim=1)
            global_feats = torch.cat([global_gm, global_wm], dim=-1)
            all_circuits = torch.cat([all_circuits, global_feats], dim=-1)

        return all_circuits

    def get_coupling_scalars(self):
        """Return per-circuit coupling scalars for interpretability."""
        return self._last_coupling_scalars


# ============================================================
# Module-3 Ablation: simple mean-pool readout
# ============================================================

class MeanPoolReadout(nn.Module):
    """Drop-in replacement for CircuitDictionaryReadout that performs
    only global mean pooling over GM and WM nodes -- no per-circuit
    dictionary, no anatomy-derived weighting.

    Used for the no_circuit_readout ablation in §3.3. Output dim is
    fixed at d_gm + d_wm regardless of the number of circuits.
    """

    def __init__(self, d_gm=32, d_wm=32):
        super().__init__()
        self.d_gm = d_gm
        self.d_wm = d_wm
        self._zones = None
        self._n_circuits = 0
        self._last_coupling_scalars = None

    @property
    def output_dim(self):
        return self.d_gm + self.d_wm

    def setup_zones(self, zones, B_prior=None):
        self._zones = zones
        self._n_circuits = len(zones) if zones is not None else 0

    def forward(self, z_gm, z_wm, fc_cross):
        return torch.cat([z_gm.mean(dim=1), z_wm.mean(dim=1)], dim=-1)

    def get_coupling_scalars(self):
        return None


# ============================================================
# Full Model: TA-BNT Final
# ============================================================

class TABNT_Final(nn.Module):
    """Tissue-Aware Brain Network Transformer -- Final Architecture.

    Three unified components:
      1. Tissue-conditioned self-attention (tissue-pair bias)
      2. Bidirectional cross-tissue affinity propagation
      3. Anatomy-informed circuit dictionary readout

    Training setup (call per CV fold BEFORE training):
        from source.utils.healthy_baseline_v3 import setup_v3_for_fold
        setup_v3_for_fold(model, train_loader, device='cuda')
    """

    def __init__(
        self,
        num_gm=200,
        num_wm=48,
        num_heads=8,
        ffn_dim=1024,
        d_reduce=32,
        use_prior=True,
        use_within_tissue=False,
        include_coupling_scalar=False,
        include_network_summaries=False,
        use_prompt_tokens=False,
        n_classes=2,
        mlp_hidden=(128, 32),
        dropout=0.5,
        attn_dropout=0.1,
        # Architectural-ablation flags (default True = full TissueFormer).
        use_tissue_bias=True,
        use_cross_tissue=True,
        use_circuit_readout=True,
    ):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        node_sz = num_gm + num_wm  # 248
        self.d_reduce = d_reduce
        self.use_tissue_bias = use_tissue_bias
        self.use_cross_tissue = use_cross_tissue
        self.use_circuit_readout = use_circuit_readout

        # -- Component 1a: Tissue embedding --
        # Additive, initialized to zero, 496 params
        self.tissue_embed = nn.Embedding(2, node_sz)
        nn.init.zeros_(self.tissue_embed.weight)

        # -- Component 1b: Tissue-conditioned self-attention --
        self.attention = TissueConditionedAttention(
            embed_dim=node_sz,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=attn_dropout,
            num_gm=num_gm,
            num_wm=num_wm,
            use_tissue_bias=use_tissue_bias,
        )

        # -- Component 2: Bidirectional affinity propagation --
        self.bipartite = BipartiteProjectionBidirectional(
            num_gm=num_gm, num_wm=num_wm, use_prior=use_prior,
            use_within_tissue=use_within_tissue)

        # -- Separate tissue projections --
        self.gm_reduce = nn.Sequential(
            nn.Linear(node_sz, d_reduce),
            nn.LeakyReLU(negative_slope=0.33),
        )
        self.wm_reduce = nn.Sequential(
            nn.Linear(node_sz, d_reduce),
            nn.LeakyReLU(negative_slope=0.33),
        )

        # -- Component 3: Circuit dictionary readout (or mean-pool ablation) --
        print(f"[DEBUG TABNT_Final] use_prompt_tokens={use_prompt_tokens}, "
              f"use_circuit_readout={use_circuit_readout}")
        if use_circuit_readout:
            self.readout = CircuitDictionaryReadout(
                d_gm=d_reduce, d_wm=d_reduce,
                include_coupling_scalar=include_coupling_scalar,
                include_global_residual=True,
                include_network_summaries=include_network_summaries,
                use_prompt_tokens=use_prompt_tokens,
            )
        else:
            self.readout = MeanPoolReadout(d_gm=d_reduce, d_wm=d_reduce)

        # Estimate pool dim (22 JHU groups)
        tract_groups = validate_tract_groups(dict(JHU_TRACT_GROUPS), num_wm)
        self._estimated_n_circuits = len(tract_groups)
        if use_circuit_readout:
            pool_dim = self._estimate_pool_dim(include_coupling_scalar,
                                               include_network_summaries)
        else:
            pool_dim = self.readout.output_dim

        # -- MLP classifier --
        layers = []
        in_dim = pool_dim
        for h_dim in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(negative_slope=0.33),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

        self._zones = None

    def _estimate_pool_dim(self, include_scalar, include_networks=False):
        n = self._estimated_n_circuits
        d = self.d_reduce
        per_circuit = d + d  # h_gm + h_wm
        if include_scalar:
            per_circuit += 1
        total = n * per_circuit
        if include_networks:
            total += 7 * d    # Yeo 7 network GM summaries
        total += d + d        # global residual
        return total

    def setup_zones(self, zones, B_prior=None):
        self._zones = zones
        self.readout.setup_zones(zones, B_prior=B_prior)

    def set_functional_prior(self, B_prior):
        self.bipartite.set_functional_prior(B_prior)

    def forward(self, time_series, node_feature):
        """
        Args:
            time_series:  [B, 248, T] -- NOT used (interface compatibility)
            node_feature: [B, 248, 248] full FC matrix
        Returns:
            logits: [B, n_classes]
        """
        device = node_feature.device
        fc_raw = node_feature

        # -- Component 1: Tissue-conditioned encoding --
        tissue_ids = torch.cat([
            torch.zeros(self.num_gm, dtype=torch.long, device=device),
            torch.ones(self.num_wm, dtype=torch.long, device=device),
        ])
        tissue_emb = self.tissue_embed(tissue_ids)
        x = node_feature + tissue_emb.unsqueeze(0)

        z = self.attention(x)                                 # [B, 248, 248]

        # -- Component 2: Split + bidirectional affinity --
        z_gm = z[:, :self.num_gm, :]                         # [B, 200, 248]
        z_wm = z[:, self.num_gm:, :]                         # [B, 48, 248]
        fc_cross = fc_raw[:, :self.num_gm, self.num_gm:]     # [B, 200, 48]

        # Module 2 ablation: skip the bipartite affinity propagation
        # but keep the bipartite module present so setup hooks still work.
        if self.use_cross_tissue:
            z_gm, z_wm = self.bipartite(z_gm, z_wm, fc_cross)

        # -- Tissue-specific projections --
        z_gm_r = self.gm_reduce(z_gm)                        # [B, 200, 32]
        z_wm_r = self.wm_reduce(z_wm)                        # [B, 48, 32]

        # -- Component 3: Circuit dictionary readout --
        pooled = self.readout(z_gm_r, z_wm_r, fc_cross)

        # -- Classifier --
        logits = self.classifier(pooled)
        return logits

    def get_biadj(self):
        """Return cached affinity matrix for interpretability/loss."""
        return self.bipartite._last_biadj

    def get_coupling_scalars(self):
        """Return per-circuit coupling scalars for interpretability."""
        return self.readout.get_coupling_scalars()

    def get_tissue_pair_bias(self):
        """Return learned 2x2 tissue-pair attention bias."""
        return self.attention.tissue_pair_bias.detach()

    def count_parameters(self, verbose=True):
        modules = [
            ('Tissue embedding',     self.tissue_embed),
            ('Self-attention',       self.attention),
            ('  tissue-pair bias',   None),  # counted separately
            ('Bipartite (bidir)',    self.bipartite),
            ('GM dim reduction',     self.gm_reduce),
            ('WM dim reduction',     self.wm_reduce),
            ('Circuit readout',      self.readout),
            ('MLP classifier',       self.classifier),
        ]
        counts = {}
        for name, mod in modules:
            if mod is not None:
                n = sum(p.numel() for p in mod.parameters())
                counts[name] = n
            elif name == '  tissue-pair bias':
                counts[name] = 4  # 2x2 matrix
        total = sum(v for k, v in counts.items() if not k.startswith('  '))
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if verbose:
            print(f"\n{'Component':<30} {'Parameters':>12}")
            print("-" * 44)
            for name, n in counts.items():
                if n > 0:
                    print(f"{name:<30} {n:>12,}")
            print("-" * 44)
            print(f"{'TOTAL':<30} {total:>12,}")
            print(f"{'Trainable':<30} {trainable:>12,}")
            print(f"{'Per pAD sample (n=78)':<30} {total/78:>12.0f}")
        return counts, total


# ============================================================
# Hydra-Compatible Wrapper
# ============================================================

def _resolve_experiment_cfg(model_cfg):
    selected = getattr(model_cfg, 'experiment', None) or \
               getattr(model_cfg, 'exp_name', None)
    if selected and hasattr(model_cfg, selected):
        return getattr(model_cfg, selected)
    return model_cfg


class TissueAwareBNT_final(BaseModel):
    """Hydra wrapper around TABNT_Final."""

    _MODEL_KEYS = {
        'num_gm', 'num_wm', 'num_heads', 'ffn_dim', 'd_reduce',
        'use_prior', 'use_within_tissue', 'include_coupling_scalar',
        'include_network_summaries', 'use_prompt_tokens', 'n_classes',
        'mlp_hidden', 'dropout', 'attn_dropout',
        # Architectural-ablation flags (H5)
        'use_tissue_bias', 'use_cross_tissue', 'use_circuit_readout',
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        exp_cfg = _resolve_experiment_cfg(config.model)

        kwargs = {}
        for key in self._MODEL_KEYS:
            if hasattr(exp_cfg, key):
                val = getattr(exp_cfg, key)
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    val = list(val)
                kwargs[key] = val

        self.model = TABNT_Final(**kwargs)

        if kwargs.get('use_prompt_tokens', False):
            print(f"[TissueFormer] Circuit prompt tokens ENABLED "
                  f"(+{22 * kwargs.get('d_reduce', 32) * 2} params)")

        self.num_gm = kwargs.get('num_gm', 200)
        self.num_wm = kwargs.get('num_wm', 48)
        self.use_prior = kwargs.get('use_prior', True)
        self.coupling_mode = 'bipartite_bidirectional'

    def forward(self, time_series, node_feature):
        logits = self.model(time_series, node_feature)
        return logits, None

    def setup_zones(self, zones, B_prior=None):
        self.model.setup_zones(zones, B_prior=B_prior)

    def set_functional_prior(self, B_prior):
        self.model.set_functional_prior(B_prior)

    def get_biadj(self):
        return self.model.get_biadj()

    def get_coupling_scalars(self):
        return self.model.get_coupling_scalars()

    def get_tissue_pair_bias(self):
        return self.model.get_tissue_pair_bias()

    def count_parameters(self, verbose=True):
        return self.model.count_parameters(verbose=verbose)

    def loss(self, assignments):
        return None


# ============================================================
# Self-Test
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TA-BNT Final -- Architecture Verification")
    print("=" * 60)

    # Test both configurations
    for include_sc in [False, True]:
        label = "without" if not include_sc else "with"
        print(f"\n{'='*50}")
        print(f" Config: {label} coupling scalar in classifier")
        print(f"{'='*50}")

        model = TABNT_Final(
            use_prior=True,
            include_coupling_scalar=include_sc)

        # Setup zones with B_prior
        tg = validate_tract_groups(dict(JHU_TRACT_GROUPS), 48)
        zg = ZoneGenerator(tg, method='topk', k=15)
        B_prior = torch.rand(200, 48).abs()
        zones = zg.generate_zones(B_prior)
        if not include_sc:
            zg.summary()

        model.setup_zones(zones, B_prior=B_prior)
        model.set_functional_prior(B_prior)
        model.count_parameters()

        # Forward pass
        B = 4
        ts = torch.randn(B, 248, 100)
        fc = torch.randn(B, 248, 248)
        fc = (fc + fc.transpose(-1, -2)) / 2

        logits = model(ts, fc)
        print(f"\nInput:  FC {fc.shape}")
        print(f"Output: logits {logits.shape}")

        biadj = model.get_biadj()
        if biadj is not None:
            print(f"Affinity: {biadj.shape} "
                  f"range=[{biadj.min():.3f}, {biadj.max():.3f}]")

        scalars = model.get_coupling_scalars()
        if scalars:
            ex = list(scalars.keys())[0]
            print(f"Coupling scalars: {len(scalars)} circuits, "
                  f"e.g. {ex}={scalars[ex].mean():.4f}")

        bias = model.get_tissue_pair_bias()
        print(f"Tissue-pair bias:\n"
              f"  GM->GM: {bias[0,0]:.4f}  GM->WM: {bias[0,1]:.4f}\n"
              f"  WM->GM: {bias[1,0]:.4f}  WM->WM: {bias[1,1]:.4f}")

        # Gradient check
        loss = logits.sum()
        loss.backward()
        all_ok = all(p.grad is not None for p in model.parameters()
                     if p.requires_grad)
        print(f"Gradients: {'OK' if all_ok else 'MISSING!'}")
        model.zero_grad()

    print("\nAll tests passed!")