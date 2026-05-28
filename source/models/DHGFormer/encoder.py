"""Transformer encoder used by DHGFormer's feature extractor.

Faithful port of ``classification/DHGFormer/model/Encoder.py``. The
encoder takes BOLD time series of shape ``(B, N, T)`` together with an
optional attention mask and returns node embeddings of shape
``(B, N, embed_dim)``.

We keep the original 4-head internal layout (the ``mask.expand(-1, 4, -1, -1)``
line in the paper's code assumes ``num_head == 4``). The default is set
in :class:`~source.models.DHGFormer.dhgformer.DHGFormer` to 4 to match.
"""

from typing import Optional

import torch
import torch.nn as nn


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiplicative attention with an optional mask pre-softmax.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Shape ``(B, H, N, d_head)``.
    mask : torch.Tensor, optional
        Shape ``(B, N, N)``. Multiplied element-wise onto the
        ``(H x N x N)`` score tensor *before* softmax, exactly as in the
        original DHGFormer implementation.

    Returns
    -------
    torch.Tensor
        Shape ``(B, N, H * d_head)``.
    """
    num_head = q.shape[1]
    seq_len = q.shape[2]

    # (B, H, N, N)
    scores = torch.matmul(q, k.permute(0, 1, 3, 2))
    scores = scores / (q.shape[-1] ** 0.5)

    if mask is not None:
        # Original code takes absolute value, adds a head dim, then
        # expands to num_head (hardcoded 4 in the source). We replicate
        # that behaviour -- the mask acts as a connectivity-weighted
        # bias on the raw scores before softmax.
        mask = torch.abs(mask).unsqueeze(1)
        mask = mask.expand(-1, num_head, -1, -1)
        scores = scores * mask

    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)

    # (B, H, N, d_head) -> (B, N, H * d_head)
    out = out.permute(0, 2, 1, 3).reshape(-1, seq_len, num_head * q.shape[3])
    return out


class _FullyConnectedOutput(nn.Module):
    """Post-attention residual-style FFN block (LayerNorm + 2-layer MLP)."""

    def __init__(self, embed_dim: int, input_dim: int) -> None:
        super().__init__()
        # We only actually use embed_dim in the network path. ``input_dim``
        # is kept in the signature purely to mirror the original file.
        del input_dim
        self.norm = nn.LayerNorm(normalized_shape=embed_dim,
                                 elementwise_affine=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.1),
            nn.Linear(32, embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(x))


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention block matching the original DHGFormer.

    Note that the projection size (32) is hardcoded in the original
    implementation, and we keep it that way to stay faithful. Only
    ``num_head`` and the output dimension are configurable.
    """

    def __init__(self, input_dim: int, num_head: int, embed_dim: int) -> None:
        super().__init__()
        # Pre-attention norm and Q/K/V projections onto a fixed 32-dim
        # space (split across heads).
        self.norm = nn.LayerNorm(normalized_shape=input_dim,
                                 elementwise_affine=True)
        self.fc_Q = nn.Linear(input_dim, 32)
        self.fc_K = nn.Linear(input_dim, 32)
        self.fc_V = nn.Linear(input_dim, 32)
        self.num_head = num_head

        # Post-attention projection into the requested embedding size.
        self.out_fc = nn.Linear(32, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, seq_len, _ = q.shape

        q = self.norm(q)
        k = self.norm(k)
        v = self.norm(v)

        q = self.fc_Q(q)
        k = self.fc_K(k)
        v = self.fc_V(v)

        # (B, N, 32) -> (B, H, N, 32/H)
        q = q.reshape(b, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
        k = k.reshape(b, seq_len, self.num_head, -1).permute(0, 2, 1, 3)
        v = v.reshape(b, seq_len, self.num_head, -1).permute(0, 2, 1, 3)

        attended = _scaled_dot_product_attention(q, k, v, mask)
        return self.dropout(self.out_fc(attended))


class _EncoderLayer(nn.Module):
    """Single attention + FFN stack -- one layer of the FCEncoder."""

    def __init__(self, input_dim: int, num_head: int, embed_dim: int) -> None:
        super().__init__()
        self.mh = _MultiHeadAttention(input_dim, num_head, embed_dim)
        self.fc = _FullyConnectedOutput(embed_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attended = self.mh(x, x, x, mask)
        return self.fc(attended)


class FCEncoder(nn.Module):
    """FC-aware transformer encoder used inside DHGFormer.

    The name "FCEncoder" is from the paper -- here "FC" stands for
    "functional connectivity", because the attention is biased by the
    Pearson correlation matrix that is passed in as ``mask``.

    Parameters
    ----------
    input_dim : int
        Length of the BOLD time-series segment fed in (equals T).
    num_head : int
        Number of attention heads (default 4 in the paper).
    embed_dim : int
        Output embedding size per ROI.
    """

    def __init__(self, input_dim: int, num_head: int, embed_dim: int) -> None:
        super().__init__()
        self.layer = _EncoderLayer(input_dim, num_head, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.layer(x, mask)
