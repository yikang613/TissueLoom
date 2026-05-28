"""InterpretableTransformerEncoder matching the Com-BrainTF paper.

This is a near-verbatim port of
``Com-BrainTF/source/models/COMTF/components/transformer_encoder.py``.
Two things differ from BNT's shared copy at ``BNT/components``:

1. ``dropout=0.3`` default (vs 0.1 in BNT) -- this is the value used in
   the published ComBrainTF results, so we keep it faithful.
2. ``average_attn_weights=False`` in the self-attention call, which
   causes ``self_attn`` to return per-head weights with shape
   ``(B, H, L, S)`` instead of the head-averaged ``(B, L, S)``.

The only non-trivial modification vs the upstream file is that we
accept an ``is_causal`` kwarg in ``_sa_block`` for forward compatibility
with PyTorch >= 2.0 (the upstream repo predates that change); the flag
is ignored since ComBrainTF does not use causal masking.
"""

from typing import Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import TransformerEncoderLayer


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    """TransformerEncoderLayer that caches per-head attention weights."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.3,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        self.attention_weights: Optional[Tensor] = None

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: Optional[bool] = False,
    ) -> Tensor:
        # ``is_causal`` is accepted only to satisfy the PyTorch >= 2.0
        # TransformerEncoderLayer calling convention; ComBrainTF never
        # masks causally.
        del is_causal
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self.attention_weights
