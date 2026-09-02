"""
Tissue-Aware Brain Network Transformer v3 (TA-BNT v3)
======================================================

A tissue-aware framework for heterogeneous brain network classification.

Design Principles:
  1. Unified encoding: single BrainnetCNN on FULL 248x248 FC
     → encoder sees ALL edges including GM-WM cross block
     → output split into Z_GM [200, d] and Z_WM [48, d]
  2. Tissue-aware coupling: zone-based cross-tissue attention + bipartite
  3. Circuit-aware pooling: aggregate by WM-defined neural circuits
  4. Parameter efficiency: ~200K total (~9x fewer than BNT's 1.75M)

Key difference from original v3 draft (which was BROKEN):
  Original: two separate encoders (GM sees 200x200, WM sees 48x48)
  → BLIND to GM-WM cross block (31% of FC = where pAD signal lives)
  → AUC ~0.53 (near chance)

  Fixed: single encoder on full 248x248 FC matrix
  → sees ALL edges including cross-tissue (same as B3/v2 encoder)
  → tissue-awareness comes from M3 coupling + M4 circuit pooling
  → no information loss, simpler architecture

Pipeline:
    Input: FC [B, 248, 248] (200 GM Schaefer + 48 WM JHU)
    |
    +-- Module 1: Tissue-Aware Input Preparation
    |     Extract fc_cross [200,48] for M3/M4
    |
    +-- Module 2: Unified BrainnetCNN Encoding
    |     Full FC [248,248] -> E2E -> E2N -> Z [B, 248, d_node]
    |     Split: Z_GM = Z[:, :200, :], Z_WM = Z[:, 200:, :]
    |
    +-- Module 3: Cross-Tissue Coupling (CORE CONTRIBUTION)
    |     3A: Bipartite B@B^T projection (5 params)
    |     3B: Zone-based cross-tissue attention (~2K params)
    |     3C: Dual path = 3A + 3B with fusion (~2K + 7 params)
    |
    +-- Module 4: Circuit-Aware Readout (CORE CONTRIBUTION)
    |     Per circuit: [h_gm, h_wm, h_gm*h_wm, coupling_scalar]
    |     + Global residual features
    |     Parameter-free pooling -> MLP classifier
    |
    +-- Output: CN vs pAD (or other binary task)

Training integration (call per CV fold BEFORE training):
    from source.utils.healthy_baseline_v3 import setup_v3_for_fold
    setup_v3_for_fold(model, train_loader, device='cuda')

References:
  - Mesulam (1990) Ann Neurol: Large-scale distributed neural networks
  - Filley & Fields (2016) J Neurophysiol: WM and cognition
  - Peer et al. (2017) J Neurosci: Functional networks in WM
  - Kawahara et al. (2017) NeuroImage: BrainnetCNN
  - Kan et al. (2022) MICCAI: BNT

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

# Schaefer 200 -> Schaefer 7 network mapping
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
# JHU 48 WM tracts -> bilateral tract group mapping
JHU_TRACT_GROUPS = {
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
# Module 1: Tissue-Aware Input Preparation
# ============================================================

class TissueAwareInput(nn.Module):
    """Extract tissue-specific submatrices from full FC. No params.

    Returns the full FC (for unified encoder) plus fc_cross
    (needed by Module 3 coupling and Module 4 circuit pooling).
    """

    def __init__(self, num_gm=200, num_wm=48):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm

    def forward(self, node_feature):
        """
        Args:
            node_feature: [B, 248, 248] full FC matrix

        Returns:
            node_feature: [B, 248, 248] full FC (passed through for encoder)
            fc_cross:     [B, 200, 48]  GM-WM cross block (for M3/M4)
        """
        g, w = self.num_gm, self.num_wm
        fc_cross = node_feature[:, :g, g:g+w]
        return node_feature, fc_cross


# ============================================================
# Module 2: Unified BrainnetCNN Encoding
# ============================================================
# KEY FIX: single encoder on full 248x248 FC, then split output.
# This ensures the encoder sees ALL edges including GM-WM cross block.

class E2EBlock(nn.Module):
    """Edge-to-Edge convolution (Kawahara et al. 2017)."""

    def __init__(self, in_planes, out_planes, d, bias=True):
        super().__init__()
        self.d = d
        self.cnn1 = nn.Conv2d(in_planes, out_planes, (1, d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, out_planes, (d, 1), bias=bias)

    def forward(self, x):
        a = self.cnn1(x)
        b = self.cnn2(x)
        return torch.cat([a] * self.d, 3) + torch.cat([b] * self.d, 2)


class BrainnetCNNEncoder(nn.Module):
    """BrainnetCNN encoder stopping at node level (E2E -> E2N).

    Unlike the original BrainnetCNN which goes to graph level (N2G -> MLP),
    we output per-node features [B, N, d_node] for Module 3.

    In v3, this runs on the FULL 248x248 FC matrix so every node's
    features incorporate connections to ALL other nodes (GM + WM).
    """

    def __init__(self, d, e2e_channels=8, d_node=32, n_e2e_layers=1):
        super().__init__()
        self.d = d
        self.d_node = d_node
        e2e_list = []
        in_ch = 1
        for i in range(n_e2e_layers):
            out_ch = e2e_channels * (2 ** i)
            e2e_list.append(E2EBlock(in_ch, out_ch, d))
            in_ch = out_ch
        self.e2e_layers = nn.ModuleList(e2e_list)
        self.e2n = nn.Conv2d(in_ch, d_node, (1, d))

    def forward(self, fc):
        """
        Args:
            fc: [B, N, N] full FC matrix (N=248 for unified encoding)
        Returns:
            node_features: [B, N, d_node]
        """
        x = fc.unsqueeze(1)    # [B, 1, N, N]
        for e2e in self.e2e_layers:
            x = F.leaky_relu(e2e(x), negative_slope=0.33)
        x = F.leaky_relu(self.e2n(x), negative_slope=0.33)
        x = x.squeeze(3).transpose(1, 2)   # [B, N, d_node]
        return x


# ============================================================
# Module 3: Cross-Tissue Coupling (CORE CONTRIBUTION)
# ============================================================

class ZoneGenerator:
    """Generate activation zones from healthy population coupling prior.

    Each zone = a WM-mediated cortical circuit: GM parcels strongly
    coupled to a tract group in healthy subjects.

    Biological basis: distributed neural networks are defined by cortical
    nodes linked through white matter pathways (Mesulam 1990).
    """

    def __init__(self, tract_groups, method='threshold', k=15,
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
        total = 0
        for name, zone in self.zones.items():
            n_gm, n_wm = len(zone['gm']), len(zone['wm'])
            total += n_gm
            print(f"{name:<20} {n_gm:>12} {n_wm:>12}")
        all_gm = set()
        for z in self.zones.values(): all_gm.update(z['gm'])
        print(f"\nUnique GM: {len(all_gm)}/200 | "
              f"Mean GM/zone: {total/len(self.zones):.1f} | "
              f"Mean zones/GM: {total/max(len(all_gm),1):.1f}")


class BipartiteProjection(nn.Module):
    """Path A: infer GM-GM connectivity via shared WM pathways (B@B^T)."""

    def __init__(self, num_gm=200, num_wm=48, use_prior=True):
        super().__init__()
        self.num_gm = num_gm
        self.use_prior = use_prior
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        if use_prior:
            self.gamma = nn.Parameter(torch.tensor(-1.0))
        self.register_buffer('B_prior', torch.zeros(num_gm, num_wm))
        self._prior_set = False
        self._last_biadj = None

    def set_functional_prior(self, B_prior):
        if not isinstance(B_prior, torch.Tensor):
            B_prior = torch.tensor(B_prior, dtype=torch.float)
        self.B_prior.copy_(B_prior / (B_prior.max() + 1e-8))
        self._prior_set = True

    def forward(self, z_gm, z_wm, fc_cross):
        D = z_gm.shape[-1]
        biadj_learned = torch.matmul(z_gm, z_wm.transpose(-2, -1)) / math.sqrt(D)
        mix = torch.sigmoid(self.beta)
        biadj = mix * fc_cross + (1 - mix) * biadj_learned
        biadj = F.relu(biadj)
        if self.use_prior and self._prior_set:
            biadj = biadj + torch.sigmoid(self.gamma) * self.B_prior.unsqueeze(0)
        self._last_biadj = biadj
        A_proj = torch.matmul(biadj, biadj.transpose(-2, -1))
        eye = torch.eye(self.num_gm, device=z_gm.device)
        A_proj = A_proj * (1 - eye)
        deg = A_proj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        A_norm = A_proj / deg
        z_gm_proj = torch.matmul(A_norm, z_gm)
        return z_gm + torch.sigmoid(self.alpha) * z_gm_proj


class CrossTissueAttention(nn.Module):
    """Path B: bidirectional cross-tissue attention within zones.

    Shared Q/K/V across all zones for parameter efficiency.
    B_raw modulates attention as subject-specific coupling bias.
    """

    def __init__(self, d_node=32, d_k=8, attn_dropout=0.1):
        super().__init__()
        self.d_k = d_k
        # Direction 1: GM <- WM
        self.W_Q1 = nn.Linear(d_node, d_k, bias=False)
        self.W_K1 = nn.Linear(d_node, d_k, bias=False)
        self.W_V1 = nn.Linear(d_node, d_k, bias=False)
        self.W_O1 = nn.Linear(d_k, d_node, bias=False)
        # Direction 2: WM <- GM
        self.W_Q2 = nn.Linear(d_node, d_k, bias=False)
        self.W_K2 = nn.Linear(d_node, d_k, bias=False)
        self.W_V2 = nn.Linear(d_node, d_k, bias=False)
        self.W_O2 = nn.Linear(d_k, d_node, bias=False)
        # Scalars
        self.gate_gm = nn.Parameter(torch.tensor(-1.0))
        self.gate_wm = nn.Parameter(torch.tensor(-1.0))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.raw_bias_scale = nn.Parameter(torch.tensor(0.5))
        self.attn_drop = nn.Dropout(attn_dropout)
        # Interpretability cache
        self._last_attn_gm_from_wm = {}
        self._last_attn_wm_from_gm = {}
        self._init_weights()

    def _init_weights(self):
        for mod in [self.W_Q1, self.W_K1, self.W_V1, self.W_O1,
                    self.W_Q2, self.W_K2, self.W_V2, self.W_O2]:
            nn.init.xavier_uniform_(mod.weight, gain=0.1)

    def _cross_attend(self, z_q, z_kv, b_raw, W_Q, W_K, W_V, W_O):
        temp = F.softplus(self.temperature) + 0.5
        bias_scale = torch.sigmoid(self.raw_bias_scale)
        Q, K, V = W_Q(z_q), W_K(z_kv), W_V(z_kv)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = attn / (math.sqrt(self.d_k) * temp)
        attn = attn + bias_scale * b_raw
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        update = W_O(torch.matmul(attn, V))
        return update, attn.detach()

    def forward(self, z_gm, z_wm, fc_cross, zones):
        B = z_gm.shape[0]
        device = z_gm.device
        gm_accum = torch.zeros_like(z_gm)
        gm_count = torch.zeros(z_gm.shape[1], device=device)
        wm_accum = torch.zeros_like(z_wm)
        wm_count = torch.zeros(z_wm.shape[1], device=device)
        self._last_attn_gm_from_wm = {}
        self._last_attn_wm_from_gm = {}

        for zone_name, zone in zones.items():
            gm_idx, wm_idx = zone['gm'], zone['wm']
            if not gm_idx or not wm_idx:
                continue
            z_gm_z = z_gm[:, gm_idx, :]
            z_wm_z = z_wm[:, wm_idx, :]
            b_raw_z = fc_cross[:, gm_idx, :][:, :, wm_idx]

            gm_up, a1 = self._cross_attend(z_gm_z, z_wm_z, b_raw_z,
                                            self.W_Q1, self.W_K1, self.W_V1, self.W_O1)
            wm_up, a2 = self._cross_attend(z_wm_z, z_gm_z, b_raw_z.transpose(-2,-1),
                                            self.W_Q2, self.W_K2, self.W_V2, self.W_O2)
            gm_accum[:, gm_idx, :] += gm_up
            gm_count[gm_idx] += 1
            wm_accum[:, wm_idx, :] += wm_up
            wm_count[wm_idx] += 1
            self._last_attn_gm_from_wm[zone_name] = a1
            self._last_attn_wm_from_gm[zone_name] = a2

        gm_count = gm_count.clamp(min=1).unsqueeze(0).unsqueeze(-1)
        wm_count = wm_count.clamp(min=1).unsqueeze(0).unsqueeze(-1)
        z_gm_out = z_gm + torch.sigmoid(self.gate_gm) * (gm_accum / gm_count)
        z_wm_out = z_wm + torch.sigmoid(self.gate_wm) * (wm_accum / wm_count)
        return z_gm_out, z_wm_out


class CrossTissueCoupling(nn.Module):
    """Module 3 wrapper: pluggable coupling with clean ablation."""

    def __init__(self, num_gm=200, num_wm=48, d_node=32, d_k=8,
                 coupling_mode='dual', use_prior=True, attn_dropout=0.1):
        super().__init__()
        self.mode = coupling_mode
        if coupling_mode in ('bipartite', 'dual'):
            self.bipartite = BipartiteProjection(num_gm, num_wm, use_prior)
        if coupling_mode in ('cross_attn', 'dual'):
            self.cross_attn = CrossTissueAttention(d_node, d_k, attn_dropout)
        if coupling_mode == 'dual':
            self.fusion_gm = nn.Parameter(torch.tensor(0.0))
            self.fusion_wm_gate = nn.Parameter(torch.tensor(-1.0))

    def set_functional_prior(self, B_prior):
        if hasattr(self, 'bipartite'):
            self.bipartite.set_functional_prior(B_prior)

    def forward(self, z_gm, z_wm, fc_cross, zones=None):
        if self.mode == 'none':
            return z_gm, z_wm
        elif self.mode == 'bipartite':
            return self.bipartite(z_gm, z_wm, fc_cross), z_wm
        elif self.mode == 'cross_attn':
            assert zones is not None
            return self.cross_attn(z_gm, z_wm, fc_cross, zones)
        elif self.mode == 'dual':
            assert zones is not None
            z_gm_bip = self.bipartite(z_gm, z_wm, fc_cross)
            z_gm_cross, z_wm_cross = self.cross_attn(z_gm, z_wm, fc_cross, zones)
            lam = torch.sigmoid(self.fusion_gm)
            z_gm_out = lam * z_gm_cross + (1 - lam) * z_gm_bip
            z_wm_out = z_wm + torch.sigmoid(self.fusion_wm_gate) * (z_wm_cross - z_wm)
            return z_gm_out, z_wm_out

    def get_biadj(self):
        if hasattr(self, 'bipartite'):
            return self.bipartite._last_biadj
        return None

    def get_attention_maps(self):
        if hasattr(self, 'cross_attn'):
            return {'gm_from_wm': self.cross_attn._last_attn_gm_from_wm,
                    'wm_from_gm': self.cross_attn._last_attn_wm_from_gm}
        return None


# ============================================================
# Module 4: Circuit-Aware Readout (CORE CONTRIBUTION)
# ============================================================

class CircuitAwarePooling(nn.Module):
    """Tissue-AWARE pooling that preserves GM, WM, and coupling per circuit.

    Unlike tissue-separate pooling (which pools GM and WM independently)
    or tissue-agnostic pooling (which lets GM dominate clusters), this
    module pools within biologically defined circuits — GM regions
    connected by WM tracts (Mesulam 1990; Filley & Fields 2016).

    Per circuit c (defined by zones from B_CN_prior):
      h_gm_c:     mean(Z_GM[zone_c_parcels])     — cortical state
      h_wm_c:     mean(Z_WM[zone_c_tracts])      — tract integrity
      h_couple_c: h_gm_c * h_wm_c                — bilinear coupling
                  (Hadamard product captures feature agreement;
                   disease disrupts this agreement)
      s_c:        mean(|fc_cross[zone_c]|)        — raw coupling strength

    Circuit representation: [h_gm, h_wm, h_couple, s_c] per circuit

    Plus global residual features to capture information from parcels
    not strongly associated with any single circuit.

    ALL operations are parameter-free. Biological structure replaces
    learned clustering.

    Args:
        num_gm: number of GM parcels
        num_wm: number of WM tracts
        d_node: node feature dimension
        include_hadamard: include h_gm * h_wm coupling features
        include_coupling_scalar: include mean |FC| coupling strength
        include_global_residual: include global mean features
    """

    def __init__(self, num_gm=200, num_wm=48, d_node=32,
                 include_hadamard=True,
                 include_coupling_scalar=True,
                 include_global_residual=True):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.d_node = d_node
        self.include_hadamard = include_hadamard
        self.include_coupling_scalar = include_coupling_scalar
        self.include_global_residual = include_global_residual

        # These will be set when zones are provided
        self._zones = None
        self._n_circuits = 0

    def setup_zones(self, zones):
        """Set zone structure from ZoneGenerator. Call per fold."""
        self._zones = zones
        self._n_circuits = len(zones)

    @property
    def output_dim(self):
        """Compute output dimensionality based on configuration."""
        n = self._n_circuits if self._n_circuits > 0 else 22  # 22 JHU groups
        d = self.d_node

        # Per circuit: h_gm (d) + h_wm (d)
        per_circuit = 2 * d
        if self.include_hadamard:
            per_circuit += d      # h_gm * h_wm (d)
        if self.include_coupling_scalar:
            per_circuit += 1      # scalar coupling

        total = n * per_circuit

        # Global residual
        if self.include_global_residual:
            total += 2 * d + 1    # global GM (d) + global WM (d) + global coupling (1)

        return total

    def forward(self, z_gm, z_wm, fc_cross):
        """
        Args:
            z_gm:     [B, 200, d_node] enriched GM features (after Module 3)
            z_wm:     [B, 48, d_node]  enriched WM features (after Module 3)
            fc_cross: [B, 200, 48]     raw GM-WM coupling from input FC

        Returns:
            circuit_features: [B, output_dim]
        """
        assert self._zones is not None, \
            "Call setup_zones() first! Use setup_v3_for_fold() from healthy_baseline_v3.py"

        B = z_gm.shape[0]
        d = self.d_node
        device = z_gm.device

        circuit_feats = []

        for zone_name, zone in self._zones.items():
            gm_idx = zone['gm']
            wm_idx = zone['wm']

            # --- GM features for this circuit ---
            if gm_idx:
                h_gm = z_gm[:, gm_idx, :].mean(dim=1)     # [B, d]
            else:
                h_gm = torch.zeros(B, d, device=device)

            # --- WM features for this circuit ---
            if wm_idx:
                h_wm = z_wm[:, wm_idx, :].mean(dim=1)     # [B, d]
            else:
                h_wm = torch.zeros(B, d, device=device)

            # Collect per-circuit features
            parts = [h_gm, h_wm]

            # --- Bilinear coupling: Hadamard product ---
            if self.include_hadamard:
                h_couple = h_gm * h_wm                     # [B, d]
                parts.append(h_couple)

            # --- Raw coupling scalar ---
            if self.include_coupling_scalar:
                if gm_idx and wm_idx:
                    coupling = fc_cross[:, gm_idx, :][:, :, wm_idx]
                    s_c = coupling.abs().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
                else:
                    s_c = torch.zeros(B, 1, device=device)
                parts.append(s_c)

            circuit_feats.append(torch.cat(parts, dim=-1))  # [B, per_circuit_dim]

        # Stack all circuits
        all_circuits = torch.cat(circuit_feats, dim=-1)     # [B, n_circuits * per_circuit]

        # --- Global residual features ---
        if self.include_global_residual:
            global_gm = z_gm.mean(dim=1)                   # [B, d]
            global_wm = z_wm.mean(dim=1)                   # [B, d]
            global_coupling = fc_cross.abs().mean(dim=(1, 2)).unsqueeze(-1)  # [B, 1]
            global_feats = torch.cat([global_gm, global_wm, global_coupling], dim=-1)
            all_circuits = torch.cat([all_circuits, global_feats], dim=-1)

        return all_circuits


class BioPooling(nn.Module):
    """Parameter-free tissue-SEPARATE pooling (ablation baseline).

    GM: Yeo 7 network mean pooling (200 -> 7)
    WM: tract group mean pooling (48 -> N_groups)

    This is the simpler alternative to CircuitAwarePooling.
    Used for ablation: compare tissue-separate vs tissue-aware readout.
    """

    def __init__(self, num_gm=200, num_wm=48,
                 yeo_networks=None, tract_groups=None):
        super().__init__()
        if yeo_networks is None:
            yeo_networks = YEO_7_NETWORKS
        self.n_networks = len(yeo_networks)
        self.network_names = list(yeo_networks.keys())

        gm_assignment = torch.full((num_gm,), 0, dtype=torch.long)
        for net_id, (name, indices) in enumerate(yeo_networks.items()):
            for idx in indices:
                if idx < num_gm:
                    gm_assignment[idx] = net_id
        self.register_buffer('gm_assignment', gm_assignment)

        # Use atlas-defined WM tract groups by default.
        if tract_groups is None:
            tract_groups = dict(JHU_TRACT_GROUPS)
        tract_groups = validate_tract_groups(tract_groups, num_wm)
        self.n_tract_groups = len(tract_groups)
        self.tract_group_names = list(tract_groups.keys())

        wm_assignment = torch.full((num_wm,), 0, dtype=torch.long)
        for grp_id, (name, indices) in enumerate(tract_groups.items()):
            for idx in indices:
                if idx < num_wm:
                    wm_assignment[idx] = grp_id
        self.register_buffer('wm_assignment', wm_assignment)

    @property
    def output_dim_factor(self):
        return self.n_networks + self.n_tract_groups

    def forward(self, z_gm, z_wm, fc_cross=None):
        B, _, d = z_gm.shape
        gm_pooled = []
        for net_id in range(self.n_networks):
            mask = (self.gm_assignment == net_id)
            if mask.any():
                gm_pooled.append(z_gm[:, mask, :].mean(dim=1))
            else:
                gm_pooled.append(torch.zeros(B, d, device=z_gm.device))
        wm_pooled = []
        for grp_id in range(self.n_tract_groups):
            mask = (self.wm_assignment == grp_id)
            if mask.any():
                wm_pooled.append(z_wm[:, mask, :].mean(dim=1))
            else:
                wm_pooled.append(torch.zeros(B, d, device=z_wm.device))
        combined = torch.cat(
            [torch.stack(gm_pooled, dim=1), torch.stack(wm_pooled, dim=1)],
            dim=1)
        return combined.reshape(B, -1)


# ============================================================
# Full Model: TA-BNT v3
# ============================================================

class TABNT_v3(nn.Module):
    """Tissue-Aware Brain Network Transformer v3.

    KEY ARCHITECTURE CHANGE from original draft:
      Old: separate GM encoder [200x200] + WM encoder [48x48]
           → BLIND to GM-WM cross block (31% of FC) → AUC ~0.53
      New: unified encoder on full [248x248] FC matrix
           → sees ALL edges → split output into Z_GM + Z_WM
           → tissue-awareness from M3 coupling + M4 circuit pooling

    Training integration — call per CV fold BEFORE training:
        from source.utils.healthy_baseline_v3 import setup_v3_for_fold
        setup_v3_for_fold(model, train_loader, device='cuda')

    Args:
        num_gm, num_wm: node counts
        e2e_channels, d_node, n_e2e_layers: BrainnetCNN encoder config
        d_k: cross-attention projection dimension
        coupling_mode: 'none', 'bipartite', 'cross_attn', 'dual'
        use_prior: use healthy population prior
        pooling_mode: 'circuit' (circuit-aware) or 'bio' (tissue-separate)
        include_hadamard: include GM*WM coupling in circuit pooling
        include_coupling_scalar: include raw FC scalar in circuit pooling
        n_classes: output classes
        mlp_hidden: MLP hidden sizes
        dropout: classifier dropout
        attn_dropout: attention dropout
    """

    def __init__(
        self,
        num_gm=200, num_wm=48,
        e2e_channels=8, d_node=32, n_e2e_layers=1,
        d_k=8,
        coupling_mode='dual', use_prior=True,
        pooling_mode='circuit',
        include_hadamard=True,
        include_coupling_scalar=True,
        n_classes=2,
        yeo_networks=None, tract_groups=None,
        mlp_hidden=(128, 32),
        dropout=0.5, attn_dropout=0.1,
    ):
        super().__init__()
        self.num_gm = num_gm
        self.num_wm = num_wm
        self.d_node = d_node
        self.coupling_mode = coupling_mode
        self.pooling_mode = pooling_mode

        # Module 1: Input preparation
        self.input_prep = TissueAwareInput(num_gm, num_wm)

        # Module 2: UNIFIED encoder on full FC (KEY FIX)
        # Single encoder on 248x248 sees ALL edges including GM-WM cross block.
        # Output is split: Z_GM = Z[:, :200, :], Z_WM = Z[:, 200:, :]
        self.encoder = BrainnetCNNEncoder(
            d=num_gm + num_wm,   # 248 for full FC
            e2e_channels=e2e_channels,
            d_node=d_node,
            n_e2e_layers=n_e2e_layers)

        # Module 3: Cross-tissue coupling
        if coupling_mode != 'none':
            self.coupling = CrossTissueCoupling(
                num_gm, num_wm, d_node, d_k,
                coupling_mode, use_prior, attn_dropout)
        else:
            self.coupling = None

        # Store early — needed by _estimate_pool_dim()
        self._zones = None
        self._include_hadamard = include_hadamard
        self._include_coupling_scalar = include_coupling_scalar

        # Module 4: Pooling
        if tract_groups is None:
            tract_groups = dict(JHU_TRACT_GROUPS)
        tract_groups = validate_tract_groups(tract_groups, num_wm)

        if pooling_mode == 'circuit':
            self.pool = CircuitAwarePooling(
                num_gm, num_wm, d_node,
                include_hadamard=include_hadamard,
                include_coupling_scalar=include_coupling_scalar)
            self._estimated_n_circuits = len(tract_groups)
            pool_dim = self._estimate_pool_dim()
        elif pooling_mode == 'bio':
            self.pool = BioPooling(num_gm, num_wm, yeo_networks, tract_groups)
            pool_dim = self.pool.output_dim_factor * d_node
        else:
            raise ValueError(f"Unknown pooling_mode: {pooling_mode}")

        # MLP classifier
        layers = []
        in_dim = pool_dim
        for h_dim in mlp_hidden:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LeakyReLU(negative_slope=0.33),
                nn.Dropout(dropout)])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def _estimate_pool_dim(self):
        """Estimate pool output dim before zones are set."""
        n = self._estimated_n_circuits
        d = self.d_node
        per_circuit = 2 * d
        if self._include_hadamard:
            per_circuit += d
        if self._include_coupling_scalar:
            per_circuit += 1
        total = n * per_circuit + 2 * d + 1  # + global residual
        return total

    def setup_zones(self, zones):
        """Set zones for Module 3 and Module 4. Call per fold."""
        self._zones = zones
        if isinstance(self.pool, CircuitAwarePooling):
            self.pool.setup_zones(zones)

    def set_functional_prior(self, B_prior):
        if self.coupling is not None:
            self.coupling.set_functional_prior(B_prior)

    def forward(self, time_series, node_feature):
        """
        Args:
            time_series:  [B, 248, T] — NOT used (interface compatibility)
            node_feature: [B, 248, 248] full FC matrix
        Returns:
            logits: [B, n_classes]
        """
        # Module 1: Extract fc_cross for M3/M4
        full_fc, fc_cross = self.input_prep(node_feature)

        # Module 2: Unified encoding on full 248x248 FC
        z_all = self.encoder(full_fc)               # [B, 248, d_node]
        z_gm = z_all[:, :self.num_gm, :]            # [B, 200, d_node]
        z_wm = z_all[:, self.num_gm:, :]            # [B, 48, d_node]

        # Module 3: Cross-tissue coupling
        if self.coupling is not None:
            z_gm, z_wm = self.coupling(z_gm, z_wm, fc_cross, self._zones)

        # Module 4: Pooling + classification
        if isinstance(self.pool, CircuitAwarePooling):
            pooled = self.pool(z_gm, z_wm, fc_cross)
        else:
            pooled = self.pool(z_gm, z_wm)

        # Classify
        logits = self.classifier(pooled)
        return logits

    def get_biadj(self):
        if self.coupling is not None:
            return self.coupling.get_biadj()
        return None

    def get_attention_maps(self):
        if self.coupling is not None:
            return self.coupling.get_attention_maps()
        return None

    def count_parameters(self, verbose=True):
        module_list = [
            ('M1: Input prep', self.input_prep),
            ('M2: Encoder', self.encoder),
            ('M3: Coupling', self.coupling),
            ('M4: Pooling', self.pool),
            ('M4: Classifier', self.classifier),
        ]
        counts = {}
        for name, module in module_list:
            if module is not None:
                n = sum(p.numel() for p in module.parameters())
                counts[name] = n
            else:
                counts[name] = 0
        total = sum(counts.values())
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
            print(f"{'Per sample (n=295)':<30} {total/295:>12.0f}")
            print(f"{'Per pAD sample (n=78)':<30} {total/78:>12.0f}")
        return counts, total


# ============================================================
# Factory + Ablation Configs
# ============================================================

ABLATION_CONFIGS = {
    # Module 3 ablation
    'v3_none': dict(coupling_mode='none', use_prior=False),
    'v3_bipartite': dict(coupling_mode='bipartite', use_prior=True),
    'v3_cross_attn': dict(coupling_mode='cross_attn', use_prior=True),
    'v3_dual': dict(coupling_mode='dual', use_prior=True),

    # Module 4 pooling ablation
    'v3_dual_bio': dict(coupling_mode='dual', use_prior=True, pooling_mode='bio'),
    'v3_dual_circuit': dict(coupling_mode='dual', use_prior=True, pooling_mode='circuit'),
    'v3_dual_circuit_no_hadamard': dict(
        coupling_mode='dual', use_prior=True, pooling_mode='circuit',
        include_hadamard=False),
    'v3_dual_circuit_no_scalar': dict(
        coupling_mode='dual', use_prior=True, pooling_mode='circuit',
        include_coupling_scalar=False),

    # Encoder capacity
    'v3_dual_light': dict(coupling_mode='dual', use_prior=True,
                          e2e_channels=4, d_node=16, d_k=4, mlp_hidden=[64, 16]),
    'v3_dual_medium': dict(coupling_mode='dual', use_prior=True,
                           e2e_channels=16, d_node=64, d_k=8),
    'v3_dual_deep': dict(coupling_mode='dual', use_prior=True, n_e2e_layers=2),
}


def build_tabnt_v3(config=None, config_name=None, **kwargs):
    """Build model from predefined config, dict, or kwargs."""
    if config_name is not None:
        cfg = ABLATION_CONFIGS[config_name].copy()
    elif config is not None:
        cfg = dict(config) if not isinstance(config, dict) else config.copy()
    else:
        cfg = {}
    cfg.update(kwargs)
    return TABNT_v3(**cfg)


def _resolve_v3_experiment_cfg(model_cfg):
    """Resolve the active experiment block from model config."""
    selected = getattr(model_cfg, 'experiment', None) or getattr(model_cfg, 'exp_name', None)
    if selected and hasattr(model_cfg, selected):
        return getattr(model_cfg, selected)
    return model_cfg


class TissueAwareBNT_v3(BaseModel):
    """Hydra-compatible wrapper around TABNT_v3.

    Accepts a full DictConfig from the training framework,
    resolves the experiment block, and delegates to TABNT_v3.
    Also returns (logits, None) to match the TABNTTrain interface.
    """

    # Fields that are model constructor args (not framework metadata)
    _MODEL_KEYS = {
        'num_gm', 'num_wm', 'e2e_channels', 'd_node', 'n_e2e_layers',
        'd_k', 'coupling_mode', 'use_prior', 'pooling_mode',
        'include_hadamard', 'include_coupling_scalar', 'n_classes',
        'yeo_networks', 'tract_groups', 'mlp_hidden', 'dropout', 'attn_dropout',
    }

    def __init__(self, config: DictConfig):
        super().__init__()
        exp_cfg = _resolve_v3_experiment_cfg(config.model)

        # Extract model kwargs from experiment config
        kwargs = {}
        for key in self._MODEL_KEYS:
            if hasattr(exp_cfg, key):
                val = getattr(exp_cfg, key)
                # Convert OmegaConf lists to plain Python lists
                if hasattr(val, '__iter__') and not isinstance(val, str):
                    val = list(val)
                kwargs[key] = val

        self.model = TABNT_v3(**kwargs)

        # Expose attributes needed by __main__.py setup
        self.coupling_mode = kwargs.get('coupling_mode', 'dual')
        self.use_prior = kwargs.get('use_prior', True)
        self.num_gm = kwargs.get('num_gm', 200)
        self.num_wm = kwargs.get('num_wm', 48)

    def forward(self, time_series, node_feature):
        logits = self.model(time_series, node_feature)
        return logits, None  # (logits, assignments=None)

    def setup_zones(self, zones):
        self.model.setup_zones(zones)

    def set_functional_prior(self, B_prior):
        self.model.set_functional_prior(B_prior)

    def get_biadj(self):
        return self.model.get_biadj()

    def get_attention_maps(self):
        return self.model.get_attention_maps()

    def count_parameters(self, verbose=True):
        return self.model.count_parameters(verbose=verbose)

    def loss(self, assignments):
        """No DEC loss for v3."""
        return None


# ============================================================
# Self-Test
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TA-BNT v3 — Unified Encoder + Tissue-Aware Coupling/Pooling")
    print("=" * 60)

    # Test all coupling modes with circuit pooling
    for mode in ['none', 'bipartite', 'cross_attn', 'dual']:
        print(f"\n{'='*50}")
        print(f" Coupling: {mode} | Pooling: circuit")
        print(f"{'='*50}")
        model = TABNT_v3(coupling_mode=mode, pooling_mode='circuit',
                         use_prior=(mode != 'none'))
        # Setup zones
        tg = validate_tract_groups(dict(JHU_TRACT_GROUPS), 48)
        zg = ZoneGenerator(tg, method='topk', k=15)
        B_prior = torch.rand(200, 48).abs()
        zones = zg.generate_zones(B_prior)
        model.setup_zones(zones)
        if mode != 'none':
            model.set_functional_prior(B_prior)
        model.count_parameters()

    # Bio pooling comparison
    print(f"\n{'='*50}")
    print(f" Coupling: dual | Pooling: bio (tissue-separate)")
    print(f"{'='*50}")
    model = TABNT_v3(coupling_mode='dual', pooling_mode='bio')
    model.count_parameters()

    # Forward pass test
    print(f"\n{'='*60}")
    print("Forward Pass Test (circuit pooling)")
    print("=" * 60)
    model = TABNT_v3(coupling_mode='dual', pooling_mode='circuit')
    tg = validate_tract_groups(dict(JHU_TRACT_GROUPS), 48)
    zg = ZoneGenerator(tg, method='topk', k=15)
    B_prior = torch.rand(200, 48).abs()
    zones = zg.generate_zones(B_prior)
    zg.summary()
    model.setup_zones(zones)
    model.set_functional_prior(B_prior)

    B = 4
    ts = torch.randn(B, 248, 100)
    fc = torch.randn(B, 248, 248)
    fc = (fc + fc.transpose(-1, -2)) / 2

    logits = model(ts, fc)
    print(f"\nInput:  node_feature {fc.shape}")
    print(f"Output: logits {logits.shape}")

    biadj = model.get_biadj()
    if biadj is not None:
        print(f"Biadj:  {biadj.shape}")
    attn = model.get_attention_maps()
    if attn:
        n_z = len(attn['gm_from_wm'])
        ex = list(attn['gm_from_wm'].keys())[0]
        print(f"Attn:   {n_z} zones, example '{ex}': {attn['gm_from_wm'][ex].shape}")

    # Gradient check
    loss = logits.sum()
    loss.backward()
    all_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients: {'OK' if all_ok else 'MISSING!'}")

    print("\nAll tests passed!")
