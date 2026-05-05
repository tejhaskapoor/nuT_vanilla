import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Set, Dict, Any, Optional, Union, List

from .nuT_components_layers import (
    Encoder_block,
)

from .nuT_components_embedding import (
    FeaturesProcessing,
    AbsolutePositionalEncoding,
    PairwiseProcessing,
    CausalityMask,
    EuclideanMask,
    IdsMask,
)

from graphnet.models.gnn.gnn import GNN # Base class for all core GNN models in graphnet.
from graphnet.models.utils import array_to_sequence # Convert `x` of shape [n, d] into a padded sequence of shape [B, L, D].

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor
   
class nuT(GNN):
    """
        Implementation of nuT model, a transformer model with pairwise interactions
        for neutrino telescopes.
    """
    def __init__(
        self,
        idx_dict: Dict,
        emb_dims: Union[List, int],
        seq_length: Union[int, None],
        emb_type: str = "nuT",
        n_features: int = 8,
        abs_position_encoding: bool = True,
        refractive_index: Union[float, None] = 1.33,
        masks: Union[List, str, None] = ['Causality', 'Euclidean', 'DUs', 'DOMs', 'PMTs'],
        mode: Union[str, None] = 'concat',
        pairwise_dims: Union[List, int] = 64,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Vanilla Transformer with pairwise attention maps.

        Args:
            seq_length: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            hidden_dim: Dimension of FFNN.
            dropout_FFNN: Dropout to be applied in MHA.
            no_hits_blocks: Number of Encoder blocks using only hit information.
            no_evt_blocks: Number of Encoder blocks including cls token, i.e. considering global event information.
        """
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__(n_features, model_dim)

        self.idx_dict = idx_dict
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.mode = mode
        self.masks = [masks] if isinstance(masks, str) else masks
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.processing = FeaturesProcessing(emb_type, model_dim, n_features)
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        # Pairwise mask modules
        self.pw_causality = CausalityMask(refractive_index) if 'Causality' in self.masks else None
        self.pw_euclidean = EuclideanMask(50) if 'Euclidean' in self.masks else None
        self.pw_du_ids = IdsMask() if 'DUs' in self.masks else None
        self.pw_dom_ids = IdsMask() if 'DOMs' in self.masks else None
        self.pw_pmt_ids = IdsMask() if 'PMTs' in self.masks else None
        
        if mode == 'concat':
            self.pw_processing = PairwiseProcessing(
                len(self.masks), 
                pairwise_dims, 
                num_heads
            )
        elif mode == 'sum':
            self.pw_processing = PairwiseProcessing(
                1, 
                pairwise_dims, 
                num_heads
            )

        self.hits_blocks = nn.Sequential(
            *[Encoder_block(
                model_dim, 
                num_heads, 
                dropout_attn, 
                hidden_dim, 
                dropout_FFNN) for _ in range(no_hits_blocks)
            ]
        )
        self.evt_blocks = nn.Sequential(
            *[Encoder_block(
                model_dim, 
                num_heads, 
                dropout_attn, 
                hidden_dim, 
                dropout_FFNN) for _ in range(no_evt_blocks)
            ]
        )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )
        
        B, L, _ = x0.shape
        
        # Class token creation
        cls_token = self.cls_token.repeat(B, 1, 1)
        
        # Need to remove trig value and ids, we do not want to process that information
        to_remove = ['trig', 'du_id', 'dom_id', 'channel_id']
        filtered_components = {key: val for key, val in self.idx_dict.items() if key not in to_remove}
        x = x0[:, :, list(filtered_components.values())]
        
        # Features processing and position encoding
        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)
        
        attn_mask = self.mode
        if self.masks and self.mode:
            masks = []
            if self.pw_causality or self.pw_euclidean:
                x_pos = x0[:, :, self.idx_dict['pos_x']].unsqueeze(-1)
                y_pos = x0[:, :, self.idx_dict['pos_y']].unsqueeze(-1)
                z_pos = x0[:, :, self.idx_dict['pos_z']].unsqueeze(-1)

                if self.pw_causality:
                    t = x0[:, :, self.idx_dict['t']].unsqueeze(-1)
                    mask_causality = self.pw_causality(torch.cat((x_pos, y_pos, z_pos, t), dim=2)).unsqueeze(1)
                    masks.append(mask_causality)

                if self.pw_euclidean:
                    mask_euclidean = self.pw_euclidean(torch.cat((x_pos, y_pos, z_pos), dim=2)).unsqueeze(1)
                    masks.append(mask_euclidean)

            if self.pw_du_ids:
                masks.append(self.pw_du_ids(x0[:, :, self.idx_dict['du_id']]).unsqueeze(1))

            if self.pw_dom_ids:
                masks.append(self.pw_dom_ids(x0[:, :, self.idx_dict['dom_id']]).unsqueeze(1))

            if self.pw_pmt_ids:
                masks.append(self.pw_pmt_ids(x0[:, :, self.idx_dict['channel_id']]).unsqueeze(1))

            masks = torch.cat(masks, dim=1)
            attn_mask = torch.sum(masks, dim=1).unsqueeze(1) if self.mode == 'sum' else masks
            attn_mask = self.pw_processing(attn_mask).view(B * self.num_heads, L, L)
            
        # Padding mask
        mask = torch.zeros(mask0.shape, dtype = attn_mask.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf
        
        if ( 
            (self.no_evt_blocks is None) or 
            (self.no_evt_blocks == 0)
        ):
            
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype = x0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            
            attn_mask = F.pad(attn_mask, (1, 0, 1, 0))

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask = mask, attn_mask = attn_mask)
                
        else:
            
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask = mask, attn_mask = attn_mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype = x0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            
            for evt_block in self.evt_blocks:
                x = evt_block(x, mask = mask)

        return x[:, 0]

class nuT_PROMETHEUS(GNN):
    """
        Implementation of nuT model, a transformer model with pairwise interactions
        for neutrino telescopes.
    """
    def __init__(
        self,
        idx_dict: Dict,
        emb_dims: Union[List, int],
        seq_length: Union[int, None],
        emb_type: str = "nuT",
        n_features: int = 8,
        abs_position_encoding: bool = True,
        refractive_index: Union[float, None] = 1.33,
        masks: Union[List, str, None] = ['Causality', 'Euclidean', 'STRING'],
        mode: Union[str, None] = 'concat',
        pairwise_dims: Union[List, int] = 64,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        ):
        """ Construct a Vanilla Transformer with pairwise attention maps.

        Args:
            seq_length: The total length of the event.
            n_features: The number of features in the input data.
            position_encoder: Wether or not, include position Fourier encoding.
            emb_dims: Embedding dimensions and/or dimension of the model.
            num_heads: Number of heads in MHA.
            dropout_attn: Dropout to be applied in MHA.
            hidden_dim: Dimension of FFNN.
            dropout_FFNN: Dropout to be applied in MHA.
            no_hits_blocks: Number of Encoder blocks using only hit information.
            no_evt_blocks: Number of Encoder blocks including cls token, i.e. considering global event information.
        """
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__(n_features, model_dim)

        self.idx_dict = idx_dict
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.mode = mode
        self.masks = [masks] if isinstance(masks, str) else masks
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.processing = FeaturesProcessing(emb_type, model_dim, n_features)
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        # Pairwise mask modules
        self.pw_causality = CausalityMask(refractive_index) if 'Causality' in self.masks else None
        self.pw_euclidean = EuclideanMask(50) if 'Euclidean' in self.masks else None
        self.pw_string_id = IdsMask() if 'STRING' in self.masks else None

        if mode == 'concat':
            self.pw_processing = PairwiseProcessing(
                len(self.masks),
                pairwise_dims,
                num_heads
            )
        elif mode == 'sum':
            self.pw_processing = PairwiseProcessing(
                1,
                pairwise_dims,
                num_heads
            )

        self.hits_blocks = nn.Sequential(
            *[Encoder_block(
                model_dim,
                num_heads,
                dropout_attn,
                hidden_dim,
                dropout_FFNN) for _ in range(no_hits_blocks)
            ]
        )
        self.evt_blocks = nn.Sequential(
            *[Encoder_block(
                model_dim,
                num_heads,
                dropout_attn,
                hidden_dim,
                dropout_FFNN) for _ in range(no_evt_blocks)
            ]
        )
    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_tocken should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data: Data) -> Tensor:
        """Apply learnable forward pass."""

        x0, mask0, evt_length = array_to_sequence(
            data.x, data.batch, padding_value=0
        )

        B, L, _ = x0.shape

        # Class token creation
        cls_token = self.cls_token.repeat(B, 1, 1)

        # Need to remove trig value and ids, we do not want to process that information
        to_remove = ['is_signal', 'string_id']
        filtered_components = {key: val for key, val in self.idx_dict.items() if key not in to_remove}
        x = x0[:, :, list(filtered_components.values())]

        # Features processing and position encoding
        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)

        attn_mask = self.mode
        if self.masks and self.mode:
            masks = []
            if self.pw_causality or self.pw_euclidean:
                x_pos = x0[:, :, self.idx_dict['sensor_pos_x']].unsqueeze(-1)
                y_pos = x0[:, :, self.idx_dict['sensor_pos_y']].unsqueeze(-1)
                z_pos = x0[:, :, self.idx_dict['sensor_pos_z']].unsqueeze(-1)

                if self.pw_causality:
                    t = x0[:, :, self.idx_dict['t']].unsqueeze(-1)
                    mask_causality = self.pw_causality(torch.cat((x_pos, y_pos, z_pos, t), dim=2)).unsqueeze(1)
                    masks.append(mask_causality)

                if self.pw_euclidean:
                    mask_euclidean = self.pw_euclidean(torch.cat((x_pos, y_pos, z_pos), dim=2)).unsqueeze(1)
                    masks.append(mask_euclidean)

            if self.pw_string_id:
                masks.append(self.pw_string_id(x0[:, :, self.idx_dict['string_id']]).unsqueeze(1))

            masks = torch.cat(masks, dim=1)
            attn_mask = torch.sum(masks, dim=1).unsqueeze(1) if self.mode == 'sum' else masks
            attn_mask = self.pw_processing(attn_mask).view(B * self.num_heads, L, L)

        # Padding mask
        mask = torch.zeros(mask0.shape, dtype = attn_mask.dtype, device = mask0.device)
        mask[~mask0] = -torch.inf

        if (
            (self.no_evt_blocks is None) or
            (self.no_evt_blocks == 0)
        ):

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype = x0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            attn_mask = F.pad(attn_mask, (1, 0, 1, 0))

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask = mask, attn_mask = attn_mask)

        else:

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask = mask, attn_mask = attn_mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype = x0.dtype, device = mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask = mask)

        return x[:, 0]
