"""
Tissue-Aware Brain Network Transformer (TA-BNT)

Extends the working BNT by adding tissue-type awareness.
Does NOT modify original BNT code - imports and extends it.

Key modification: Adds tissue embedding to node identity,
allowing the model to distinguish GM vs WM nodes.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

# Import from existing BNT codebase (adjust path as needed)
from ..BNT.bnt import BrainNetworkTransformer, TransPoolingEncoder
from ..base import BaseModel


class TissueAwareBNT(BaseModel):
    """
    Tissue-Aware Brain Network Transformer.
    
    This is BNT + additive tissue embedding.
    
    The tissue embedding is ADDED to node_identity, so:
    - GM nodes get: node_identity[i] + tissue_embed[0]
    - WM nodes get: node_identity[i] + tissue_embed[1]
    
    Architecture is identical to BNT - same parameter count except
    for the small tissue embedding (2 × pos_embed_dim = 32 params).
    
    Expected performance: Should match or exceed BNT's ~0.60 AUC.
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        # === Tissue configuration ===
        self.num_gm_nodes = getattr(config.model, 'num_gm_nodes', 200)
        self.num_wm_nodes = getattr(config.model, 'num_wm_nodes', 48)
        
        # === COPIED FROM BNT (unchanged) ===
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), 
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # === NEW: Tissue embedding (same dim as pos_embed) ===
        self.tissue_embed = nn.Embedding(2, config.model.pos_embed_dim)
        nn.init.zeros_(self.tissue_embed.weight)  # Start at zero (neutral)
        # === END NEW ===

        sizes = config.model.sizes.copy()  # Copy to avoid modifying original
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
                    project_assignment=config.model.project_assignment
                )
            )

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

    def forward(self, time_seires: torch.tensor, node_feature: torch.tensor):
        bz, _, _ = node_feature.shape
        device = node_feature.device

        if self.pos_encoding == 'identity':
            # === MODIFIED: Add tissue embedding to node identity ===
            tissue_ids = torch.cat([
                torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
                torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
            ])
            tissue_emb = self.tissue_embed(tissue_ids)  # (248, pos_embed_dim)
            
            # Tissue-modulated node identity
            combined_identity = self.node_identity + tissue_emb
            pos_emb = combined_identity.unsqueeze(0).expand(bz, -1, -1)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)
            # === END MODIFIED ===

        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature), assignments

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def loss(self, assignments):
        """Compute KL loss for DEC clustering."""
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
        """Get cluster centers from all pooling layers."""
        centers = []
        for atten in self.attention_list:
            if atten.is_pooling_enabled():
                centers.append(atten.dec.get_cluster_centers())
        return centers

    def get_tissue_embedding(self):
        """Get learned tissue embeddings for interpretability."""
        return {
            'GM': self.tissue_embed.weight[0].detach().cpu().numpy(),
            'WM': self.tissue_embed.weight[1].detach().cpu().numpy(),
            'difference': (self.tissue_embed.weight[1] - self.tissue_embed.weight[0]).detach().cpu().numpy()
        }


# ============================================================
# Alternative: Concatenation version (more expressive but larger)
# ============================================================

class TissueAwareBNT_Concat(BaseModel):
    """
    TA-BNT with concatenated tissue embedding.
    
    Instead of adding tissue embedding to node_identity,
    this version concatenates it, increasing forward_dim.
    
    More expressive but slightly more parameters.
    Use this if the additive version doesn't improve over BNT.
    """

    def __init__(self, config: DictConfig):
        super().__init__()

        self.num_gm_nodes = getattr(config.model, 'num_gm_nodes', 200)
        self.num_wm_nodes = getattr(config.model, 'num_wm_nodes', 48)
        tissue_embed_dim = getattr(config.model, 'tissue_embed_dim', 8)
        
        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(
                torch.zeros(config.dataset.node_sz, config.model.pos_embed_dim), 
                requires_grad=True
            )
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        # Tissue embedding (separate dimension, concatenated)
        self.tissue_embed = nn.Embedding(2, tissue_embed_dim)
        nn.init.normal_(self.tissue_embed.weight, std=0.02)
        forward_dim += tissue_embed_dim  # Increase dimension

        sizes = config.model.sizes.copy()
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
                    project_assignment=config.model.project_assignment
                )
            )

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

    def forward(self, time_seires: torch.tensor, node_feature: torch.tensor):
        bz, _, _ = node_feature.shape
        device = node_feature.device

        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.unsqueeze(0).expand(bz, -1, -1)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        # Concatenate tissue embedding
        tissue_ids = torch.cat([
            torch.zeros(self.num_gm_nodes, dtype=torch.long, device=device),
            torch.ones(self.num_wm_nodes, dtype=torch.long, device=device)
        ])
        tissue_emb = self.tissue_embed(tissue_ids).unsqueeze(0).expand(bz, -1, -1)
        node_feature = torch.cat([node_feature, tissue_emb], dim=-1)

        assignments = []
        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)
        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature), assignments

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def loss(self, assignments):
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all