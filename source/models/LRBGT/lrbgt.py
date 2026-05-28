"""LRBGT: Local Random-walk Brain Graph Transformer.

Port of ``alter/models/LRBGT/lrbgt.py`` from the stand-alone ALTER repo
(https://github.com/yushuowiki/ALTER). The original class was called
``BrainNetworkTransformer`` -- we rename it to :class:`LRBGT` here so
it does not shadow BNT's existing class of that name. The architecture
is otherwise byte-for-byte faithful to the paper/repo.

Key ideas preserved from the original:

* Optional **RRWP** (Relative Random Walk Probabilities) positional
  encoding is appended to each ROI's row of the Pearson matrix before
  attention. When ``pos_encoding: rrwp`` is set in the YAML, every
  sample's FC matrix is row-normalised (``D^{-1} A``) and its first
  ``pos_embed_dim`` walk-step self-probabilities are concatenated to
  the node features.
* ``identity`` positional encoding (a learned per-node bias) is also
  supported, exactly as in BNT's transformer baseline.
* The body is a stack of :class:`TransPoolingEncoder` blocks (attention
  + optional DEC pooling) with ``nhead=4``, reusing the
  ``InterpretableTransformerEncoder`` and ``DEC`` utilities that BNT
  already ships under ``source/models/BNT/``.

Shapes expected by :meth:`LRBGT.forward`:

``time_seires``: ``(B, N, T)`` (unused here, kept for BaseModel parity).
``node_feature``: ``(B, N, N)`` Pearson matrix.
Returns: logits of shape ``(B, out_dim)`` with ``out_dim == 2`` in
the default config.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

# Reuse BNT's already-ported shared components -- behaviourally identical
# to the copies that live inside the ALTER repo.
from ..BNT.components import InterpretableTransformerEncoder
from ..BNT.ptdec import DEC
from ..base import BaseModel


# ---------------------------------------------------------------------------
# RRWP positional-encoding helpers.
# ---------------------------------------------------------------------------
def add_full_rrwp(data: torch.Tensor, walk_length: int) -> torch.Tensor:
    """Batched wrapper around :func:`add_every_rrwp`.

    Parameters
    ----------
    data : torch.Tensor
        Shape ``(B, N, N)`` Pearson/connectivity matrices.
    walk_length : int
        Number of random-walk steps used to build the positional
        embedding (equivalent to ``pos_embed_dim`` in the config).

    Returns
    -------
    torch.Tensor
        Shape ``(B, N, walk_length)`` positional embeddings.
    """
    pes = []
    for ids in range(data.shape[0]):
        # ``.squeeze()`` is a no-op for (N, N) inputs but preserves the
        # original code path.
        dt = data[ids].squeeze()
        pe = add_every_rrwp(dt, walk_length)
        pes.append(pe)
    return torch.stack(pes)


def add_every_rrwp(
    data: torch.Tensor,
    walk_length: int = 8,
    add_identity: bool = True,
    spd: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Relative-random-walk positional encoding for a single FC matrix.

    Builds ``[I, A, A^2, ..., A^{walk_length-1}]`` where ``A`` is the
    row-normalised adjacency, then returns the main-diagonal stack
    (i.e. the probability of returning to each node after ``k`` steps).

    Notes
    -----
    * ``spd`` is accepted for API compatibility with the original
      ALTER code path but is not used.
    * Rows whose sum is zero produce NaNs after normalisation; these
      are zeroed out, matching the original implementation.
    """
    # Build a COO edge list from non-zero FC entries, then go back to
    # dense for the matrix power. This mirrors the reference code
    # exactly, which deliberately round-trips through sparse.
    edge_index = torch.column_stack(torch.where(data > 0.0)).T.contiguous()

    device = edge_index.device
    num_nodes = data.shape[0]
    edge_weight = data[edge_index[0], edge_index[1]]
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj.to_dense()

    # Row-stochastic normalisation: A <- D^{-1} A.
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float("inf")] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list: List[torch.Tensor] = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float, device=device))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    # Stack walk-steps along the last axis and take the diagonal so we
    # end up with a (N, walk_length) matrix of return-probabilities.
    pe = torch.stack(pe_list, dim=-1).cuda()  # (N, N, K)
    abs_pe = pe.diagonal().transpose(0, 1)  # (N, K)
    return abs_pe


# ---------------------------------------------------------------------------
# Transformer block with optional DEC pooling.
# ---------------------------------------------------------------------------
class TransPoolingEncoder(nn.Module):
    """Transformer encoder with an optional DEC pooling head.

    Input:  ``(B, input_node_num,  input_feature_size)``
    Output: ``(B, output_node_num, input_feature_size)`` (pooling)
            ``(B, input_node_num,  input_feature_size)`` (no pooling)
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
    ) -> None:
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(
            d_model=input_feature_size,
            nhead=4,
            dim_feedforward=hidden_size,
            batch_first=True,
        )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            # Bottleneck MLP used by DEC to compute soft cluster
            # assignments over tokens.
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

    def is_pooling_enabled(self) -> bool:
        return self.pooling

    def forward(self, x: torch.Tensor):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


# ---------------------------------------------------------------------------
# The main model.
# ---------------------------------------------------------------------------
class LRBGT(BaseModel):
    """Local Random-walk Brain Graph Transformer (renamed ALTER LRBGT).

    The original repository called this class ``BrainNetworkTransformer``;
    we rename it to :class:`LRBGT` so the model factory can dispatch
    unambiguously.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim
        if self.pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim),
                requires_grad=True,
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == "rrwp":
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim

        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling
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

    def forward(
        self,
        time_seires: torch.Tensor,
        node_feature: torch.Tensor,
    ) -> torch.Tensor:
        bz, _, _ = node_feature.shape

        if self.pos_encoding == "identity":
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        if self.pos_encoding == "rrwp":
            pos_emb = add_full_rrwp(node_feature, self.pos_embed_dim)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """Return the DEC cluster centres of the final pooling layer."""
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """KL-divergence DEC clustering loss over all pooling layers.

        The stock BNT trainer does not call this -- it uses CE only --
        so LRBGT trains with CE alone by default. Custom trainers that
        want the full paper objective should invoke ``model.loss(...)``
        after each forward pass, with ``assignments`` collected from
        the pooling-enabled encoders.
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
