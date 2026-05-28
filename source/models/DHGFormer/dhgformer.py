"""DHGFormer -- Dynamic Hierarchical Graph Transformer (MICCAI 2025).

Faithful port of the model from ``classification/DHGFormer/model/``,
wrapped to follow the BrainNetworkTransformer convention:

* subclasses :class:`source.models.base.BaseModel`
* takes a single :class:`omegaconf.DictConfig` in ``__init__``
* matches the shared ``forward(time_seires, node_feature)`` signature

Architecture summary
--------------------
1. ROIs are reordered according to a pickled ``node_clus_map`` so that
   nodes belonging to the same functional subnetwork are contiguous.
2. An :class:`~source.models.DHGFormer.encoder.FCEncoder` consumes the
   ROI-reordered BOLD time series, biased by the Pearson FC matrix used
   as an attention mask, and emits one embedding per ROI.
3. A graph generator (``product`` or ``linear``) turns those embeddings
   into three adjacency tensors: intra-subnetwork, inter-subnetwork (a
   small K x K coarse graph where K = number of subnetworks), and the
   full adjacency. The full adjacency is stored on the module for the
   optional auxiliary losses described below.
4. A hierarchical three-layer graph convolution (``CrossGCNPredictor``)
   alternates intra-subnetwork propagation on reordered node features
   with inter-subnetwork propagation on subnetwork-averaged features,
   finishing with a small MLP classifier.

Differences from the original
-----------------------------
* **Return signature.** The original ``forward`` returns
  ``(logits, full_adjacency, edge_variance)``. To comply with the
  :class:`BaseModel` contract and the default
  ``source/training/training.py`` loop, this port returns only logits.
  The last forward pass's auxiliary tensors are retained as
  ``self._last_full_adjacency`` and ``self._last_edge_variance``, and
  the paper's group-loss + sparsity-loss hook is exposed via
  :meth:`DHGFormer.loss`. A custom training script is required to
  actually add those terms to the total loss -- the stock trainer will
  only use the classification CE loss, which still reproduces the
  "no-aux-loss" ablation.
* **Configurable subnetwork partition.** The original hardcodes
  ``[41, 70, 91, 110, 130, 137, 158, 200]`` (the Yeo-7 subnetwork
  boundaries of a 200-ROI atlas). Here both ``subnetwork_ends`` and the
  number of ROIs come from config, so re-using the model on a different
  atlas only requires updating the YAML and ``node_clus_map`` pickle.
* **Pickle path.** The original opens ``./node_clus_map.pickle``
  relative to the CWD, which is brittle. We default to the pickle that
  lives next to this file, and allow an override via
  ``model.node_clus_map_path`` in the config.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..base import BaseModel
from .encoder import FCEncoder


# ---------------------------------------------------------------------------
# Graph generators -- turn (B, N, embed_dim) node embeddings into adjacency
# tensors. Two options: ``product`` (dot-product similarity + subnetwork
# pooling) and ``linear`` (learned sender/receiver projection).
# ---------------------------------------------------------------------------


class _CrossEmbed2GraphByProduct(nn.Module):
    """Dot-product similarity with explicit subnetwork masking."""

    def __init__(self, input_dim: int, roi_num: int) -> None:
        super().__init__()
        # Arguments kept for API parity with the original; the module
        # itself has no learnable parameters.
        del input_dim, roi_num

    @staticmethod
    def _get_subnetwork_matrix(
        adjacency: torch.Tensor,
        subnetwork_ends: Sequence[int],
    ) -> torch.Tensor:
        """Mean-pool ``adjacency`` blocks into a ``K x K`` coarse graph."""
        starts = [0] + list(subnetwork_ends[:-1])
        num_sub = len(subnetwork_ends)
        batch = adjacency.shape[0]

        coarse = torch.zeros(
            (batch, num_sub, num_sub),
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        # Symmetric fill: compute the upper triangle (inclusive) and
        # mirror it.
        for i in range(num_sub):
            for j in range(i, num_sub):
                block = adjacency[
                    :,
                    starts[i]:subnetwork_ends[i],
                    starts[j]:subnetwork_ends[j],
                ]
                mean_strength = block.mean(dim=(1, 2))
                coarse[:, i, j] = mean_strength
                coarse[:, j, i] = mean_strength
        return coarse

    def forward(
        self,
        embeddings: torch.Tensor,
        subnetwork_ends: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Full pairwise similarity matrix: (B, N, N).
        adjacency = torch.einsum("ijk,ipk->ijp", embeddings, embeddings)

        # Build an (N, N) boolean mask that is True only inside the
        # subnetwork diagonal blocks -- this is the "intra" mask.
        roi_count = embeddings.shape[1]
        intra_mask = torch.zeros(
            (roi_count, roi_count), dtype=torch.bool, device=embeddings.device
        )
        start = 0
        for end in subnetwork_ends:
            intra_mask[start:end, start:end] = True
            start = end
        intra_adj = adjacency * intra_mask.unsqueeze(0)

        # Coarse inter-subnetwork matrix (K x K), used for message
        # passing between communities.
        inter_adj = self._get_subnetwork_matrix(adjacency, subnetwork_ends)

        # Add a trailing channel dim so the downstream code can index
        # ``[..., 0]`` uniformly regardless of graph generator.
        return (
            intra_adj.unsqueeze(-1),
            inter_adj.unsqueeze(-1),
            adjacency.unsqueeze(-1),
        )


class _Embed2GraphByLinear(nn.Module):
    """Learned edge scoring via sender/receiver concatenation.

    Only a single ``full_adjacency`` tensor is produced; ``intra`` and
    ``inter`` are then derived from it in :class:`DHGFormer.forward`
    exactly as in the ``product`` branch, so the downstream predictor
    sees a consistent API.
    """

    def __init__(self, input_dim: int, roi_num: int) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(input_dim * 2, input_dim)
        self.edge_predictor = nn.Linear(input_dim, 1)

        # Precompute one-hot sender/receiver indices on CPU; move them
        # to the correct device lazily in forward() to avoid requiring
        # CUDA at construction time (unlike the original which did
        # .cuda() in __init__).
        off_diag = np.ones([roi_num, roi_num])
        rows, cols = np.where(off_diag)
        eye = np.eye(roi_num, dtype=np.float32)
        self.register_buffer("receiver_matrix", torch.from_numpy(eye[rows]),
                             persistent=False)
        self.register_buffer("sender_matrix", torch.from_numpy(eye[cols]),
                             persistent=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        subnetwork_ends: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ``_CrossEmbed2GraphByProduct.forward`` ignores ``subnetwork_ends``
        # when building the full adjacency -- we do the same here, then
        # derive the intra/inter tensors below.
        batch_size, region_count, _ = embeddings.shape
        receivers = torch.matmul(self.receiver_matrix, embeddings)
        senders = torch.matmul(self.sender_matrix, embeddings)
        edge_feats = torch.cat([senders, receivers], dim=2)
        edge_feats = torch.relu(self.feature_proj(edge_feats))
        edge_scores = torch.relu(self.edge_predictor(edge_feats))
        full_adj = edge_scores.reshape(
            batch_size, region_count, region_count
        )

        # Derive intra/inter from the learned adjacency.
        intra_mask = torch.zeros(
            (region_count, region_count),
            dtype=torch.bool,
            device=embeddings.device,
        )
        start = 0
        for end in subnetwork_ends:
            intra_mask[start:end, start:end] = True
            start = end
        intra_adj = full_adj * intra_mask.unsqueeze(0)
        inter_adj = _CrossEmbed2GraphByProduct._get_subnetwork_matrix(
            full_adj, subnetwork_ends
        )
        return (
            intra_adj.unsqueeze(-1),
            inter_adj.unsqueeze(-1),
            full_adj.unsqueeze(-1),
        )


# ---------------------------------------------------------------------------
# Hierarchical graph-convolution predictor.
# ---------------------------------------------------------------------------


class _CrossGCNPredictor(nn.Module):
    """Three-layer hierarchical GCN with a small classifier on top."""

    def __init__(
        self,
        node_input_dim: int,
        roi_num: int,
        subnetwork_ends: Sequence[int],
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.roi_num = roi_num
        # Stored as a plain Python list (not a buffer) because the
        # subnetwork partition is a structural property, not a tensor.
        self.subnetwork_ends: List[int] = list(subnetwork_ends)

        # Layer-1 projection: node_input_dim (== N in typical use) -> N.
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(roi_num, roi_num),
        )
        self.bn1 = nn.BatchNorm1d(roi_num)

        # Layer-2 keeps dimensionality.
        self.gcn1 = nn.Sequential(
            nn.Linear(roi_num, roi_num),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = nn.BatchNorm1d(roi_num)

        # Layer-3 compresses each ROI to an 8-dim vector.
        self.gcn2 = nn.Sequential(
            nn.Linear(roi_num, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = nn.BatchNorm1d(roi_num)

        # Final classifier (flattens 8 x N per sample).
        self.classifier = nn.Sequential(
            nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, num_classes),
        )

    # ----- subnetwork pooling helpers --------------------------------------

    def _average_subnetwork_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool node features within each subnetwork."""
        batch_size, _, feature_dim = features.shape
        starts = [0] + self.subnetwork_ends[:-1]
        num_sub = len(self.subnetwork_ends)

        sub_features = torch.zeros(
            (batch_size, num_sub, feature_dim),
            device=features.device,
            dtype=features.dtype,
        )
        for i, (s, e) in enumerate(zip(starts, self.subnetwork_ends)):
            sub_features[:, i, :] = features[:, s:e, :].mean(dim=1)
        return sub_features

    def _propagate_subnetwork_features(
        self,
        subnetwork_features: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast pooled subnetwork features back onto individual ROIs."""
        starts = [0] + self.subnetwork_ends[:-1]
        propagated = torch.zeros_like(node_features)
        for i, (s, e) in enumerate(zip(starts, self.subnetwork_ends)):
            expanded = (
                subnetwork_features[:, i, :]
                .unsqueeze(1)
                .expand(-1, e - s, -1)
            )
            propagated[:, s:e, :] = expanded
        # Average the original and propagated features so each ROI keeps
        # some of its own identity (same 0.5/0.5 mix as the original).
        return (node_features + propagated) / 2

    # ----- forward ---------------------------------------------------------

    def _one_layer(
        self,
        x: torch.Tensor,
        intra_adj: torch.Tensor,
        inter_adj: torch.Tensor,
        conv: nn.Module,
        bn: nn.Module,
    ) -> torch.Tensor:
        """One hierarchical propagation + conv + batchnorm cycle."""
        # Intra-subnetwork message passing: multiply per-ROI features by
        # the masked intra-adjacency (element-wise contract over j=p).
        intra_features = torch.einsum("ijk,ijp->ijp", intra_adj, x)
        # Inter-subnetwork message passing: pool to K subnetworks, push
        # through the coarse K x K graph, then broadcast back.
        sub_features = self._average_subnetwork_features(x)
        sub_features = torch.einsum("ijk,ijp->ijp", inter_adj, sub_features)
        x = self._propagate_subnetwork_features(sub_features, intra_features)

        x = conv(x)
        batch_size = x.shape[0]
        x = x.reshape(batch_size * self.roi_num, -1)
        x = bn(x)
        x = x.reshape(batch_size, self.roi_num, -1)
        return x

    def forward(
        self,
        adjacency: torch.Tensor,
        intra_adj: torch.Tensor,
        inter_adj: torch.Tensor,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        # ``adjacency`` is unused here (kept in the signature only to
        # match the original DHGFormer code). The three GCN layers below
        # only look at intra/inter adjacency and the node features.
        del adjacency

        x = self._one_layer(node_features, intra_adj, inter_adj,
                            self.gcn, self.bn1)
        x = self._one_layer(x, intra_adj, inter_adj, self.gcn1, self.bn2)

        # Final layer: same propagation pattern but the batchnorm is
        # applied *after* the convolution (matching the original).
        intra_features = torch.einsum("ijk,ijp->ijp", intra_adj, x)
        sub_features = self._average_subnetwork_features(x)
        sub_features = torch.einsum("ijk,ijp->ijp", inter_adj, sub_features)
        x = self._propagate_subnetwork_features(sub_features, intra_features)
        x = self.gcn2(x)
        x = self.bn3(x)

        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Top-level DHGFormer module.
# ---------------------------------------------------------------------------


class DHGFormer(BaseModel):
    """Dynamic Hierarchical Graph Transformer (MICCAI 2025) ported to BNT."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        model_cfg = config.model
        dataset_cfg = config.dataset

        # ----- Sizes ------------------------------------------------------
        self.roi_num = int(dataset_cfg.node_sz)
        time_series_len = int(dataset_cfg.timeseries_sz)
        node_feature_dim = int(dataset_cfg.node_feature_sz)
        embedding_size = int(model_cfg.embedding_size)
        num_head = int(getattr(model_cfg, "num_head", 4))

        # Subnetwork partition: list of cumulative right-edges. Must sum
        # to roi_num. Default is the Yeo-7 200-ROI partition used in the
        # paper.
        default_ends = [41, 70, 91, 110, 130, 137, 158, 200]
        self.subnetwork_ends: List[int] = list(
            getattr(model_cfg, "subnetwork_ends", default_ends)
        )
        if self.subnetwork_ends[-1] != self.roi_num:
            raise ValueError(
                "subnetwork_ends[-1] must equal dataset.node_sz "
                f"(got {self.subnetwork_ends[-1]} vs {self.roi_num}). "
                "Update conf/model/dhgformer.yaml (or supply a matching "
                "node_clus_map.pickle for your atlas)."
            )

        # ----- Feature extractor -----------------------------------------
        extractor_type = getattr(model_cfg, "extractor_type", "transformer")
        if extractor_type == "transformer":
            self.feature_extractor = FCEncoder(
                input_dim=time_series_len,
                num_head=num_head,
                embed_dim=embedding_size,
            )
        else:
            raise ValueError(
                f"Unknown extractor_type={extractor_type!r}. "
                "Only 'transformer' is currently ported."
            )

        # ----- Graph generator -------------------------------------------
        self.graph_generation = getattr(
            model_cfg, "graph_generation", "product"
        )
        if self.graph_generation == "product":
            self.graph_generator = _CrossEmbed2GraphByProduct(
                embedding_size, roi_num=self.roi_num
            )
        elif self.graph_generation == "linear":
            self.graph_generator = _Embed2GraphByLinear(
                embedding_size, roi_num=self.roi_num
            )
        else:
            raise ValueError(
                f"Unknown graph_generation={self.graph_generation!r}. "
                "Must be 'product' or 'linear'."
            )

        # ----- Hierarchical GCN predictor --------------------------------
        self.predictor = _CrossGCNPredictor(
            node_input_dim=node_feature_dim,
            roi_num=self.roi_num,
            subnetwork_ends=self.subnetwork_ends,
            num_classes=int(getattr(model_cfg, "out_dim", 2)),
        )

        # ----- Load node -> cluster mapping ------------------------------
        # The pickle is a dict whose *keys* are the desired ROI order
        # (usually 0..N-1 permuted so that same-cluster ROIs are
        # contiguous). We register it as a plain LongTensor buffer so it
        # travels with the module to GPU and is saved in state_dict.
        pickle_path = getattr(
            model_cfg,
            "node_clus_map_path",
            str(Path(__file__).parent / "node_clus_map.pickle"),
        )
        with open(pickle_path, "rb") as f:
            node_cluster_map = pickle.load(f)
        cluster_order = list(node_cluster_map.keys())
        if len(cluster_order) != self.roi_num:
            raise ValueError(
                f"node_clus_map has {len(cluster_order)} ROIs but "
                f"dataset.node_sz is {self.roi_num}. Provide a pickle "
                "matching your atlas via model.node_clus_map_path."
            )
        self.register_buffer(
            "cluster_order",
            torch.as_tensor(cluster_order, dtype=torch.long),
            persistent=False,
        )

        # ----- Auxiliary-loss bookkeeping --------------------------------
        # Weight for the sparsity-on-adjacency regulariser (L1 of the
        # learned full adjacency). The group-loss (mixup_cluster_loss)
        # needs mixup labels which are only available inside the training
        # loop, so it is computed by the trainer, not here.
        self.sparsity_loss_weight = float(
            getattr(model_cfg, "sparsity_loss_weight", 1.0e-4)
        )

        # Tensors captured on the most recent forward pass, exposed so
        # an outer training loop can compute the paper's group +
        # sparsity losses. Initialised to ``None`` so the module is
        # safe to introspect before the first forward call.
        self._last_full_adjacency: Optional[torch.Tensor] = None
        self._last_edge_variance: Optional[torch.Tensor] = None

    # ----- Helpers --------------------------------------------------------

    def _reorder_time_series(self, time_series: torch.Tensor) -> torch.Tensor:
        """Reorder ROIs in a ``(B, N, T)`` time-series tensor."""
        return time_series.index_select(1, self.cluster_order)

    def _reorder_fc_matrix(self, fc: torch.Tensor) -> torch.Tensor:
        """Reorder rows *and* columns of a ``(B, N, N)`` FC matrix."""
        fc = fc.index_select(1, self.cluster_order)
        fc = fc.index_select(2, self.cluster_order)
        return fc

    # ----- Forward --------------------------------------------------------

    def forward(
        self,
        time_seires: torch.Tensor,
        node_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification logits.

        Parameters
        ----------
        time_seires : torch.Tensor
            Shape ``(B, N, T)`` -- BOLD time series from the BNT
            dataloader.
        node_feature : torch.Tensor
            Shape ``(B, N, N)`` -- Pearson FC matrix.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        # 1. Reorder so that ROIs of the same Yeo subnetwork sit in
        #    contiguous blocks matching ``self.subnetwork_ends``.
        time_seires = self._reorder_time_series(time_seires)
        node_feature = self._reorder_fc_matrix(node_feature)

        # 2. Transformer feature extractor. ``mask=node_feature`` turns
        #    attention into FC-aware attention.
        embeddings = self.feature_extractor(time_seires, node_feature)

        # The authors softmax the per-ROI embeddings along the feature
        # axis -- this normalises them onto the simplex before the
        # dot-product graph generator.
        embeddings = F.softmax(embeddings, dim=-1)

        # 3. Build adjacency tensors. Each has a trailing channel dim.
        intra_adj, inter_adj, full_adj = self.graph_generator(
            embeddings, self.subnetwork_ends
        )
        # Drop the trailing channel dim (all three were added purely for
        # a consistent API across graph generators).
        full_adj = full_adj[..., 0]
        intra_adj = intra_adj[..., 0]
        inter_adj = inter_adj[..., 0]

        # 4. Edge-variance monitoring term (mean variance across all
        #    edges per sample). Used by the paper's logging and kept for
        #    parity; not part of the default loss.
        batch_size = full_adj.shape[0]
        edge_variance = torch.mean(
            torch.var(full_adj.reshape(batch_size, -1), dim=1)
        )

        # Stash auxiliary outputs for model.loss() / downstream access.
        self._last_full_adjacency = full_adj
        self._last_edge_variance = edge_variance

        # 5. Hierarchical GCN predictor.
        return self.predictor(full_adj, intra_adj, inter_adj, node_feature)

    # ----- Optional auxiliary loss hook -----------------------------------

    def loss(
        self,
        y_a: Optional[torch.Tensor] = None,
        y_b: Optional[torch.Tensor] = None,
        lam: float = 1.0,
        intra_weight: float = 2.0,
    ) -> torch.Tensor:
        """Compute the paper's auxiliary regularisers.

        This is meant to be called **after** a forward pass, optionally
        by a custom training loop that wants to reproduce the full
        DHGFormer training objective:

        .. math:: L_{total} = L_{CE} + L_{group} + w_{sparse} L_{sparse}

        where ``L_group`` is ``mixup_cluster_loss`` from the original
        repo and ``L_sparse`` is the L1 norm of the learned adjacency.
        The returned tensor is ``L_group + w_{sparse} L_{sparse}``.

        Parameters
        ----------
        y_a, y_b : torch.Tensor, optional
            Mixup label pairs for the batch, shape ``(B,)``. If
            ``lam`` equals 1 (no mixup), pass ``y_a = y_b = labels``.
            When either is ``None``, the group-loss term is skipped.
        lam : float, default 1.0
            Mixup interpolation coefficient.
        intra_weight : float, default 2.0
            Weight of the centre-distance term inside
            ``mixup_cluster_loss`` (matches the paper's default).

        Notes
        -----
        Running ``DHGFormer`` through ``source/training/training.py`` as
        shipped will *not* use this method -- that trainer only applies
        cross-entropy. Writing a DHGFormer-specific trainer (mirroring
        ``source/training/FBNettraining.py``) is the recommended way to
        bring these terms back in.
        """
        if self._last_full_adjacency is None:
            raise RuntimeError(
                "DHGFormer.loss() called before any forward pass."
            )

        full_adj = self._last_full_adjacency

        # Sparsity: simple L1 penalty on the learned adjacency.
        sparsity_term = self.sparsity_loss_weight * torch.norm(full_adj, p=1)

        # Group loss: optional, needs mixup labels.
        group_term = full_adj.new_zeros(())
        if y_a is not None and y_b is not None:
            group_term = _mixup_cluster_loss(
                full_adj,
                y_a=y_a,
                y_b=y_b,
                lam=float(lam),
                intra_weight=float(intra_weight),
            )

        return group_term + sparsity_term

    def last_edge_variance(self) -> Optional[torch.Tensor]:
        """Return the edge-variance statistic from the last forward pass."""
        return self._last_edge_variance


# ---------------------------------------------------------------------------
# Helper: verbatim port of ``util/loss.py::mixup_cluster_loss`` so this
# module is fully self-contained.
# ---------------------------------------------------------------------------


def _mixup_cluster_loss(
    matrixs: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    intra_weight: float = 2.0,
) -> torch.Tensor:
    """Class-centred L1 clustering loss with centre-distance regulariser.

    Faithful port of ``util/loss.py::mixup_cluster_loss`` from the
    original DHGFormer repo.
    """
    y_1 = lam * y_a.float() + (1 - lam) * y_b.float()
    y_0 = 1 - y_1

    batch_size, roi_num, _ = matrixs.shape
    flat = matrixs.reshape(batch_size, -1)
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = flat.new_zeros(())

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, flat) / sum_0
        diff_0 = torch.norm(flat - center_0, p=1, dim=1)
        loss = loss + torch.matmul(y_0, diff_0) / (
            sum_0 * roi_num * roi_num
        )
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, flat) / sum_1
        diff_1 = torch.norm(flat - center_1, p=1, dim=1)
        loss = loss + torch.matmul(y_1, diff_1) / (
            sum_1 * roi_num * roi_num
        )
    if sum_0 > 0 and sum_1 > 0:
        loss = loss + intra_weight * (
            1 - torch.norm(center_0 - center_1, p=1) / (roi_num * roi_num)
        )
    return loss
