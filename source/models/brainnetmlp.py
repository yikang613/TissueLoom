"""BrainNetMLP ported into the BrainNetworkTransformer framework.

Faithful port of the model from
``classification/BrainNetMLP/model/brainnetmlp.py`` (EMA4MICCAI 2025),
wrapped to follow this repo's conventions:

* subclasses :class:`source.models.base.BaseModel`
* takes a single :class:`omegaconf.DictConfig` in ``__init__``
* matches the shared ``forward(time_seires, node_feature)`` signature

The architecture has two parallel projections that are concatenated
before a small classification head:

1. **Spatial stream** -- flattens the upper triangle of the functional
   connectivity matrix and projects it with a single linear layer.
2. **Spectral stream** -- takes the real FFT of the BOLD time series,
   drops the DC component and any frequencies above index ``k``, then
   averages over the remaining frequency bins and projects them.

Important shape note on the time series:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The original BrainNetMLP expects ``ts`` with shape ``(B, T, N)``
(time-first, as produced by its ``normalize`` + transpose in
``utilis.load_data``). The BrainNetworkTransformer dataloader passes
time series as ``(B, N, T)`` (ROI-first, consistent with FBNETGEN and
BNT). We therefore transpose inside ``forward`` before handing the
tensor to the FFT path. No normalisation is performed here -- if you
want to reproduce the paper's spectrum features you should apply
per-node z-scoring upstream (as BrainNetMLP's ``utilis.normalize``
does).
"""

from typing import Sequence

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseModel


class BrainNetMLP(BaseModel):
    """Two-stream MLP for fMRI brain-network classification."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        model_cfg = config.model
        dataset_cfg = config.dataset

        # ----- Input / output dimensions ----------------------------------
        # ``in_dim`` is the number of ROIs (N), ``out_dim`` the number of
        # classes. We read N from dataset.node_sz to stay consistent with
        # the rest of the framework; ``out_dim`` defaults to 2 (binary)
        # but can be overridden via the model config.
        in_dim = int(dataset_cfg.node_sz)
        out_dim = int(getattr(model_cfg, "out_dim", 2))

        # Hidden dimensions of the two streams: (spectral, spatial).
        # The original paper default is (32, 32).
        hidden_dim: Sequence[int] = tuple(model_cfg.hidden_dim)
        assert len(hidden_dim) == 2, "hidden_dim must be a 2-tuple"

        # Dropout rates after each projection: (spectral_drop, spatial_drop).
        drop_rate: Sequence[float] = tuple(model_cfg.drop_rate)
        assert len(drop_rate) == 2, "drop_rate must be a 2-tuple"

        # Low-pass FFT cutoff index. The original model keeps bins
        # ``[1:k]`` (i.e. drops the DC component and everything above k).
        self.k = int(model_cfg.k)

        # Whether to apply BatchNorm1d on the concatenated embedding.
        use_norm = bool(int(model_cfg.norm))

        # ----- Precomputed triangular indices -----------------------------
        # We precompute row/col indices for the upper triangle (including
        # the diagonal) exactly as in the original code. Registering them
        # as buffers means they travel with the module to GPU.
        rows, cols = torch.triu_indices(in_dim, in_dim, offset=0)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)

        # Length of the flattened upper-triangle vector: (N+1) * N / 2.
        # Matches the original's ``(in_dim+1)*(in_dim//2)`` expression,
        # which is only exact for even N. We use the exact formula to be
        # safe for odd N as well.
        num_upper = (in_dim * (in_dim + 1)) // 2

        # ----- Two projection streams -------------------------------------
        # Spectral stream: project mean FFT magnitudes (one value per ROI)
        # through a Linear into hidden_dim[0].
        self.freq_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0]),
            nn.Dropout(drop_rate[0]),
        )

        # Spatial stream: project flattened upper-triangle of the FC
        # matrix into hidden_dim[1].
        self.spatial_proj = nn.Sequential(
            nn.Linear(num_upper, hidden_dim[1]),
            nn.Dropout(drop_rate[1]),
        )

        # Optional BatchNorm between the concatenation and the decoder.
        self.norm = (
            nn.BatchNorm1d(hidden_dim[0] + hidden_dim[1])
            if use_norm
            else nn.Identity()
        )

        # ----- Classification head ----------------------------------------
        # GELU + single linear, matching the original's decoder.
        self.decoder = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim[0] + hidden_dim[1], out_dim),
        )

    def forward(
        self,
        time_seires: torch.Tensor,
        node_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification logits.

        Parameters
        ----------
        time_seires : torch.Tensor
            BOLD time series of shape ``(B, N, T)`` -- ROI-first, as
            produced by the BNT dataloader. We transpose internally so
            that FFT is taken along the time axis.
        node_feature : torch.Tensor
            Pearson functional-connectivity matrix of shape
            ``(B, N, N)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, out_dim)``.
        """
        # --- Spectral stream ----------------------------------------------
        # BrainNetMLP's FFT path expects (B, T, N), but we receive
        # (B, N, T). Transpose and make contiguous so torch.fft is happy.
        ts = time_seires.transpose(1, 2).contiguous()

        # Real FFT along the time dimension (dim=1 after transpose).
        x_f = torch.fft.rfft(ts, dim=1)

        # Magnitude spectrum, then keep only bins [1, k) -- i.e. low-pass
        # filter that also drops the DC bin. Guard against k exceeding the
        # number of available bins (T//2 + 1) since that would break the
        # linear projection's input size otherwise.
        f_t = torch.abs(x_f)
        k = min(self.k, f_t.shape[1])
        f_t = f_t[:, 1:k, :]

        # Project each (low-pass) spectrum per ROI, then average over the
        # remaining frequency bins. Output shape: (B, hidden_dim[0]).
        f_t = self.freq_proj(f_t).mean(dim=1)

        # --- Spatial stream -----------------------------------------------
        # Pull out the upper triangle of the FC matrix as a flat vector.
        # (B, N, N) -> (B, num_upper)
        x = node_feature[:, self.rows, self.cols]
        x = x.flatten(start_dim=1)

        # Project to hidden_dim[1].
        x = self.spatial_proj(x)

        # --- Fuse and classify --------------------------------------------
        x = torch.cat((x, f_t), dim=1)
        x = self.norm(x)
        return self.decoder(x)
