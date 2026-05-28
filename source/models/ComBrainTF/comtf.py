"""ComBrainTF: Community-aware Brain Transformer.

Port of ``Com-BrainTF/source/models/COMTF/comtf.py`` (ubc-tea/Com-BrainTF)
into the BrainNetworkTransformer convention. The architecture stays
byte-for-byte faithful to the paper, but two small changes keep the
port self-contained and atlas-aware:

1. ``node_clus_map.pickle`` is loaded relative to this file (not CWD),
   with an optional absolute override via ``config.model.node_clus_map_path``.
2. The cumulative subnetwork boundaries (8 Yeo-7 networks at
   ``[41, 70, 91, 110, 130, 137, 158, 200]`` in the paper) are exposed
   as ``config.model.subnetwork_ends``. The constructor validates that
   the pickle has ``node_sz`` entries and that the ends close at
   ``node_sz``. Non-8-subnetwork atlases are currently unsupported
   because the model hardcodes 8 learnable class tokens and an
   ``8 * d -> 1024 -> d`` MLP after the local transformer.

Shapes expected by :meth:`ComBrainTF.forward`:

``time_seires``: ``(B, N, T)`` (unused, kept for BaseModel parity).
``node_feature``: ``(B, N, N)`` Pearson matrix.
Returns: logits of shape ``(B, 2)``.
"""

import pickle
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base import BaseModel
# Reuse BNT's already-ported DEC helpers. The ComBrainTF implementation
# of ``ptdec`` is identical to BNT's.
from ..BNT.ptdec import DEC
from .components import InterpretableTransformerEncoder


# ---------------------------------------------------------------------------
# Transformer block with optional DEC pooling and optional local-CLS token.
# ---------------------------------------------------------------------------
class TransPoolingEncoder(nn.Module):
    """Transformer encoder with optional DEC pooling and optional local CLS.

    When ``local_transformer=True`` the block is used as the *local*
    subnetwork encoder: it owns 8 learnable class tokens (one per Yeo-7
    subnetwork), prepends the relevant token to the input, and returns
    the updated tokens alongside the transformer-refined node features
    plus the pooled CLS output.

    When ``local_transformer=False`` it is used as the standard global
    encoder, optionally with DEC pooling.
    """

    def __init__(
        self,
        input_feature_size: int,
        input_node_num: int,
        hidden_size: int,
        output_node_num: int,
        pooling: bool = True,
        orthogonal: bool = True,
        freeze_center: bool = False,
        project_assignment: bool = True,
        nHead: int = 4,
        local_transformer: bool = False,
    ) -> None:
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=nHead,
            dim_feedforward=hidden_size,
            batch_first=True,
        )

        self.local_transformer = local_transformer
        # Local-CLS-mode blocks do not pool -- pooling is only defined for
        # the single global encoder that follows them.
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling

        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(
                    input_feature_size * input_node_num, encoder_hidden_size
                ),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(
                    encoder_hidden_size, input_feature_size * input_node_num
                ),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=input_feature_size,
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
            )

        if local_transformer:
            # Eight learnable class tokens, one per Yeo-7 subnetwork.
            # We do not ``.cuda()`` them here -- ``model.cuda()`` on the
            # full ComBrainTF will move them along with every other
            # parameter, which is the portable way to put tensors on GPU.
            self.class_token = nn.ParameterList()
            for _ in range(8):
                self.class_token.append(
                    nn.Parameter(
                        torch.Tensor(1, input_feature_size), requires_grad=True
                    )
                )
            self.reset_parameters(local_transformer)

        # MLP used in the original to fuse the 8 local CLS tokens -- kept
        # here for faithfulness even though ``ComBrainTF`` has its own
        # fusion MLP.
        self.mlp = nn.Sequential(
            nn.Linear(8 * input_feature_size, 1024),
            nn.Linear(1024, input_feature_size),
            nn.ReLU(),
        )

    def reset_parameters(self, local_transformer: bool = False) -> None:
        if local_transformer:
            for i in range(len(self.class_token)):
                self.class_token[i] = nn.init.xavier_normal_(
                    self.class_token[i]
                )

    def is_pooling_enabled(self) -> bool:
        return self.pooling

    def forward(self, x: torch.Tensor, cluster_num: int = -1):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            # Prepend the subnetwork-specific class token to the node
            # features so the transformer can pool into it.
            class_token = self.class_token[cluster_num]
            class_token = class_token.repeat(bz, 1, 1)
            x = torch.cat((class_token, x), dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            # The global transformer gets an extra CLS token from the
            # wrapping ComBrainTF module, so indexing out [:, 0, :] /
            # [:, 1:, :] mirrors the paper's implementation.
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
            return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


# ---------------------------------------------------------------------------
# Main model.
# ---------------------------------------------------------------------------
class ComBrainTF(BaseModel):
    """Community-aware Brain Transformer (ComBrainTF)."""

    # Default Yeo-7 cumulative ends on a 200-ROI Schaefer atlas.
    _DEFAULT_SUBNETWORK_ENDS = [41, 70, 91, 110, 130, 137, 158, 200]

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(
                    config.dataset.node_sz, config.model.pos_embed_dim
                ),
                requires_grad=True,
            )
            forward_dim = (
                config.dataset.node_sz + config.model.pos_embed_dim
            )
            nn.init.kaiming_normal_(self.node_identity)

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        # ------------------------------------------------------------------
        # Local (subnetwork-wise) transformer.
        # ------------------------------------------------------------------
        self.local_transformer = TransPoolingEncoder(
            input_feature_size=forward_dim,
            input_node_num=in_sizes[1],
            hidden_size=1024,
            output_node_num=sizes[1],
            pooling=False,
            orthogonal=config.model.orthogonal,
            freeze_center=config.model.freeze_center,
            project_assignment=config.model.project_assignment,
            nHead=config.model.nhead,
            local_transformer=True,
        )

        # ------------------------------------------------------------------
        # Global transformer stack.
        # ------------------------------------------------------------------
        if config.model.num_MHSA == 1:
            self.attention_list.append(
                TransPoolingEncoder(
                    input_feature_size=forward_dim,
                    input_node_num=in_sizes[1],
                    hidden_size=1024,
                    output_node_num=sizes[1],
                    pooling=do_pooling[1],
                    orthogonal=config.model.orthogonal,
                    freeze_center=config.model.freeze_center,
                    project_assignment=config.model.project_assignment,
                    nHead=config.model.nhead,
                    local_transformer=False,
                )
            )
        else:
            for index, size in enumerate(sizes):
                self.attention_list.append(
                    TransPoolingEncoder(
                        input_feature_size=forward_dim,
                        input_node_num=in_sizes[index],
                        hidden_size=1024,
                        output_node_num=size,
                        pooling=do_pooling[index],
                        orthogonal=config.model.orthogonal,
                        freeze_center=config.model.freeze_center,
                        project_assignment=config.model.project_assignment,
                        nHead=config.model.nhead,
                        local_transformer=False,
                    )
                )

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2),
        )

        self.assignMat = None
        # CLS-fusion MLP: takes the 8 stacked per-subnetwork CLS tokens
        # and produces a single fused CLS token of width `forward_dim`.
        self.mlp = nn.Sequential(
            nn.Linear(8 * forward_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, forward_dim),
            nn.LeakyReLU(),
        )

        # ------------------------------------------------------------------
        # Load the ROI -> subnetwork mapping. Prefer an absolute path
        # passed via YAML; otherwise fall back to the pickle bundled next
        # to this file.
        # ------------------------------------------------------------------
        pickle_path_override = getattr(config.model, "node_clus_map_path", None)
        if pickle_path_override:
            pickle_path = Path(pickle_path_override)
        else:
            pickle_path = Path(__file__).parent / "node_clus_map.pickle"
        with open(pickle_path, "rb") as handle:
            self.node_clus_map = pickle.load(handle)

        # ------------------------------------------------------------------
        # Resolve the cumulative subnetwork-end indices. Must have
        # exactly 8 entries (one per learnable class token) and end at
        # the atlas size.
        # ------------------------------------------------------------------
        ends_cfg = getattr(config.model, "subnetwork_ends", None)
        if ends_cfg is None:
            self.node_rearranged_len: List[int] = list(
                self._DEFAULT_SUBNETWORK_ENDS
            )
        else:
            self.node_rearranged_len = list(ends_cfg)

        node_sz = int(config.dataset.node_sz)
        if len(self.node_rearranged_len) != 8:
            raise ValueError(
                "ComBrainTF hardcodes 8 learnable class tokens and an "
                "8*d fusion MLP, so subnetwork_ends must have exactly "
                f"8 entries; got {self.node_rearranged_len}"
            )
        if self.node_rearranged_len[-1] != node_sz:
            raise ValueError(
                f"subnetwork_ends={self.node_rearranged_len} must close "
                f"at dataset.node_sz={node_sz}"
            )
        if len(self.node_clus_map) != node_sz:
            raise ValueError(
                f"node_clus_map.pickle has {len(self.node_clus_map)} "
                f"entries but dataset.node_sz={node_sz}"
            )

    # ------------------------------------------------------------------
    # Forward helpers.
    # ------------------------------------------------------------------
    def rearrange_node_feature(
        self,
        node_feature_rearranged: torch.Tensor,
        node_feature: torch.Tensor,
        rearranged_indices,
    ) -> torch.Tensor:
        """Reorder ROIs so that each subnetwork is a contiguous block.

        ``node_clus_map`` is a dict ``{orig_roi: cluster_id}`` keyed in
        the order we want to rearrange to. The keys of the dict define
        the permutation.
        """
        # First permute rows, then columns -- the result is still
        # (B, N, N) but with ROIs grouped by subnetwork along both axes.
        node_feature_rearranged = node_feature[:, rearranged_indices, :]
        node_feature_rearranged = node_feature_rearranged[:, :, rearranged_indices]
        return node_feature_rearranged

    def forward(
        self,
        time_seires: torch.Tensor,
        node_feature: torch.Tensor,
    ) -> torch.Tensor:
        bz, _, _ = node_feature.shape

        if self.pos_encoding == "identity":
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []
        attn_weights = []

        # Rearrange ROIs into subnetwork-ordered blocks.
        node_feature_rearranged = self.rearrange_node_feature(
            None, node_feature, list(self.node_clus_map.keys())
        )

        # Run the local transformer on each subnetwork slice with its
        # matching class token, writing the refined features back in
        # place and collecting the per-subnetwork CLS tokens.
        local_class_tokens = []
        prev_end = 0
        for cluster_num, end in enumerate(self.node_rearranged_len):
            slab = node_feature_rearranged[:, prev_end:end, :]
            slab_out, _, cls = self.local_transformer(slab, cluster_num=cluster_num)
            node_feature_rearranged[:, prev_end:end, :] = slab_out
            local_class_tokens.append(cls)
            prev_end = end

        node_feature = node_feature_rearranged
        class_token = torch.cat(local_class_tokens, dim=1)  # (B, 8, d)
        class_token = class_token.reshape((bz, -1))  # (B, 8*d)
        class_token = self.mlp(class_token)  # (B, d)
        class_token = class_token.reshape((bz, 1, -1))
        node_feature = torch.cat((class_token, node_feature), dim=1)

        if self.num_MHSA == 1:
            node_feature, assign_mat, cls_token = self.attention_list[0](
                node_feature
            )
            assignments.append(assign_mat)
            attn_weights.append(self.attention_list[0].get_attention_weights())
        else:
            for atten in self.attention_list:
                node_feature, _, cls_token = atten(node_feature)
                attn_weights.append(atten.get_attention_weights())

        # Cache the DEC assignment matrix so callers can query it via
        # ``model.get_assign_mat()`` after a forward pass -- keeps the
        # BaseModel forward() signature clean.
        self.assignMat = assignments[0] if assignments else None

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """Cluster centres of the final DEC-pooled encoder, if any."""
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """KL DEC clustering loss over pooling-enabled encoders.

        Not called by the stock BNT trainer (which uses CE only) -- kept
        here so a custom trainer can run the full paper objective.
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list)
        )
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None
        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
