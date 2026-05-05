"""Optimized version of nuT model with performance improvements."""

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

from graphnet.models.gnn.gnn import GNN
from graphnet.models.utils import array_to_sequence

from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import Tensor


class OptimizedCausalityMask(nn.Module):
    """Optimized causality mask using torch.cdist for distance calculations."""

    def __init__(
        self,
        refractive_index: float = 1.33,
        scaling_xyz: float = 1.0,
        scaling_t: float = 1.0e-9,
    ):
        super().__init__()
        self.refractive_index = refractive_index
        self.c = 299792458.0
        self.v = self.c / self.refractive_index
        self.scaling_xyz = scaling_xyz
        self.scaling_t = scaling_t

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optimized distance calculations."""
        pos = x[:, :, :3] * self.scaling_xyz
        time = x[:, :, 3] * self.scaling_t

        # Use torch.cdist for more efficient distance computation
        pos_diff = torch.cdist(pos, pos, p=2)  # [B, L, L]
        time_diff = (time.unsqueeze(2) - time.unsqueeze(1)) * self.v

        spacetime_interval = pos_diff.pow(2) - time_diff.pow(2)
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval) + 1e-8  # Add epsilon for numerical stability
        )
        return four_distance.clamp_(-4, 4)  # In-place operation


class OptimizedEuclideanMask(nn.Module):
    """Optimized euclidean mask using torch.cdist."""

    def __init__(self, max_distance: float = 50.0):
        super().__init__()
        self.max_distance = max_distance

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optimized distance calculation."""
        # Use torch.cdist for more efficient Euclidean distance
        euclidean_distance = torch.cdist(x, x, p=2)  # [B, L, L]
        return euclidean_distance.clamp_(0, self.max_distance)  # In-place operation


class OptimizedIdsMask(nn.Module):
    """Optimized ID mask with pre-computation for static IDs."""

    def __init__(self, cache_static: bool = True):
        super().__init__()
        self.cache_static = cache_static
        self._cached_mask = None
        self._cached_ids = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional caching."""
        if self.cache_static and self._cached_mask is not None:
            if torch.equal(x, self._cached_ids):
                return self._cached_mask

        # Vectorized comparison using broadcasting
        mask = (x.unsqueeze(2) == x.unsqueeze(1)).float()

        if self.cache_static:
            self._cached_mask = mask
            self._cached_ids = x.clone()

        return mask


class OptimizedEncoder_block(nn.Module):
    """Optimized encoder block with disabled attention weights and potential for gradient checkpointing."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        use_checkpoint: bool = False,  # For potential gradient checkpointing
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(dim)

        # Use GELU approximation for faster computation
        self.FFNN = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),  # Faster approximation
            nn.Dropout(dropout_FFNN),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_FFNN),
        )

    def _forward_impl(self, x, mask, attn_mask=None):
        """Implementation that can be checkpointed."""
        z = self.ln_1(x)
        # Optimized attention: don't compute weights we don't use
        attn_output = self.self_attention(
            z,
            z,
            z,
            key_padding_mask=mask,
            attn_mask=attn_mask,
            need_weights=False,  # Performance optimization
            average_attn_weights=False,
        )[0]
        x = x + attn_output
        y = self.ln_2(x)
        x = x + self.FFNN(y)
        return x

    def forward(self, x, mask, attn_mask=None):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing for memory efficiency
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, attn_mask
            )
        else:
            return self._forward_impl(x, mask, attn_mask)


class OptimizedFeaturesProcessing(nn.Module):
    """Optimized features processing with bug fix and efficient operations."""

    def __init__(
        self,
        emb_dims: Union[List, int],
        emb_type: str = "nuT",
        n_features: int = 6,
    ):
        super().__init__()

        if emb_type == "nuT":
            assert isinstance(emb_dims, int), (
                f"Only one embedding dimension is possible while {emb_dims} was provided"
            )
            self.emb = nn.Linear(n_features, emb_dims)
            self.model_dim = emb_dims
        elif emb_type == "Kaggle":
            assert isinstance(emb_dims, list) and len(emb_dims) == 2, (
                f"Only two embedding dimensions are possible while {emb_dims} was provided"
            )
            self.model_dim = emb_dims[-1]
            module_list = []
            module_list.extend(
                [
                    nn.Linear(n_features, emb_dims[0]),  # Bug fix: was emb_dim[0]
                    nn.LayerNorm(emb_dims[0]),
                    nn.GELU(approximate="tanh"),  # Faster approximation
                    nn.Linear(emb_dims[0], emb_dims[1]),
                ]
            )
            self.emb = nn.Sequential(*module_list)
        elif emb_type == "ParticleTransformer":
            if isinstance(emb_dims, int):
                emb_dims = [emb_dims]
            self.model_dim = emb_dims[-1]
            module_list = []
            for emb_dim in emb_dims:
                module_list.extend(
                    [
                        nn.LayerNorm(n_features),
                        nn.Linear(n_features, emb_dim),
                        nn.GELU(approximate="tanh"),  # Faster approximation
                    ]
                )
                n_features = emb_dim
            self.emb = nn.Sequential(*module_list)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.model_dim)


class nuT_optimized(GNN):
    """
    Optimized version of nuT model with performance improvements:
    1. Efficient attention (disabled weight computation)
    2. Optimized mask computations using torch.cdist
    3. In-place operations and caching where applicable
    """

    def __init__(
        self,
        idx_dict: Dict,
        emb_dims: Union[List, int],
        seq_length: Union[int, None],
        emb_type: str = "nuT",
        n_features: int = 6,
        abs_position_encoding: bool = True,
        refractive_index: Union[float, None] = 1.33,
        masks: Union[List, str, None] = ["Causality", "Euclidean", "STRING"],
        mode: Union[str, None] = "concat",
        pairwise_dims: Union[List, int] = 64,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        use_gradient_checkpointing: bool = False,  # New parameter
        cache_static_masks: bool = True,  # New parameter
    ):
        """Construct optimized Transformer."""
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__(n_features, model_dim)

        self.idx_dict = idx_dict
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.mode = mode
        self.masks = [masks] if isinstance(masks, str) else masks
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cache_static_masks = cache_static_masks
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Use optimized processing
        self.processing = OptimizedFeaturesProcessing(model_dim, emb_type, n_features)
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, seq_length or 300)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        # Optimized pairwise mask modules
        self.pw_causality = (
            OptimizedCausalityMask(refractive_index or 1.33)
            if self.masks and "Causality" in self.masks
            else None
        )
        self.pw_euclidean = (
            OptimizedEuclideanMask(50)
            if self.masks and "Euclidean" in self.masks
            else None
        )
        self.pw_string_id = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "STRING" in self.masks
            else None
        )
        self.pw_euclidean = (
            OptimizedEuclideanMask(50)
            if self.masks and "Euclidean" in self.masks
            else None
        )
        self.pw_du_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "DUs" in self.masks
            else None
        )
        self.pw_dom_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "DOMs" in self.masks
            else None
        )
        self.pw_pmt_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "PMTs" in self.masks
            else None
        )

        if mode == "concat":
            self.pw_processing = PairwiseProcessing(
                len(self.masks or []), pairwise_dims, num_heads
            )
        elif mode == "sum":
            self.pw_processing = PairwiseProcessing(1, pairwise_dims, num_heads)

        # Use optimized encoder blocks
        self.hits_blocks = nn.ModuleList(
            [
                OptimizedEncoder_block(
                    model_dim,
                    num_heads,
                    dropout_attn,
                    hidden_dim,
                    dropout_FFNN,
                    use_checkpoint=use_gradient_checkpointing,
                )
                for _ in range(no_hits_blocks)
            ]
        )

        self.evt_blocks = nn.ModuleList(
            [
                OptimizedEncoder_block(
                    model_dim,
                    num_heads,
                    dropout_attn,
                    hidden_dim,
                    dropout_FFNN,
                    use_checkpoint=use_gradient_checkpointing,
                )
                for _ in range(no_evt_blocks)
            ]
        )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_token should not be subject to weight decay during training."""
        return {"cls_token"}

    def _compute_masks_optimized(self, x0: Tensor, B: int, L: int) -> Optional[Tensor]:
        """Optimized mask computation with vectorized operations."""
        if not self.masks or not self.mode:
            return None

        masks = []

        # Pre-extract position data once
        if self.pw_causality or self.pw_euclidean:
            x_pos = x0[:, :, self.idx_dict["sensor_pos_x"]].unsqueeze(-1)
            y_pos = x0[:, :, self.idx_dict["sensor_pos_y"]].unsqueeze(-1)
            z_pos = x0[:, :, self.idx_dict["sensor_pos_z"]].unsqueeze(-1)
            positions = torch.cat((x_pos, y_pos, z_pos), dim=2)

            if self.pw_causality:
                t = x0[:, :, self.idx_dict["t"]].unsqueeze(-1)
                spacetime_data = torch.cat((positions, t), dim=2)
                mask_causality = self.pw_causality(spacetime_data).unsqueeze(1)
                masks.append(mask_causality)

            if self.pw_euclidean:
                mask_euclidean = self.pw_euclidean(positions).unsqueeze(1)
                masks.append(mask_euclidean)

        # Process ID masks efficiently
        if self.pw_string_id:
            masks.append(
                self.pw_string_id(x0[:, :, self.idx_dict["string_id"]]).unsqueeze(1)
            )

        # Efficient concatenation and processing
        if masks:
            masks = torch.cat(masks, dim=1)
            attn_mask = (
                torch.sum(masks, dim=1).unsqueeze(1) if self.mode == "sum" else masks
            )
            attn_mask = self.pw_processing(attn_mask).view(B * self.num_heads, L, L)
            return attn_mask

        return None

    def forward(self, data: Data) -> Tensor:
        """Optimized forward pass."""
        x0, mask0, evt_length = array_to_sequence(data.x, data.batch, padding_value=0)

        B, L, _ = x0.shape

        # Class token creation
        cls_token = self.cls_token.repeat(B, 1, 1)

        # Feature filtering (optimized)
        to_remove = ["is_signal", "string_id"]
        filtered_components = {
            key: val for key, val in self.idx_dict.items() if key not in to_remove
        }
        x = x0[:, :, list(filtered_components.values())]

        # Features processing and position encoding
        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)

        # Optimized mask computation
        attn_mask = self._compute_masks_optimized(x0, B, L)

        # Padding mask (vectorized)
        mask = torch.zeros(
            mask0.shape,
            dtype=attn_mask.dtype if attn_mask is not None else torch.float32,
            device=mask0.device,
        )
        mask[~mask0] = -torch.inf

        if (self.no_evt_blocks is None) or (self.no_evt_blocks == 0):
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (1, 0, 1, 0))

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask=attn_mask)

        else:
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask=attn_mask)

            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask=mask)

        return x[:, 0]


# Optional: Try to enable torch.compile for additional optimization (PyTorch 2.0+)
#try:
#    nuT_optimized = torch.compile(nuT_optimized, mode="reduce-overhead")
#except AttributeError:
#    pass  # torch.compile not available
