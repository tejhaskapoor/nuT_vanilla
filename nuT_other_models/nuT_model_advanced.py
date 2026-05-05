"""Further optimized version of nuT model with architecture, I/O, and bug fix improvements."""

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


# Import optimized mask classes from the first optimization file
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


class FlashAttentionEncoder(nn.Module):
    """Encoder block with Flash Attention support for better performance."""

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        use_flash_attention: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self.use_checkpoint = use_checkpoint

        self.ln_1 = nn.LayerNorm(dim)

        # Try to use Flash Attention if available
        if use_flash_attention and hasattr(F, "scaled_dot_product_attention"):
            # Use PyTorch's built-in Flash Attention
            self.attention_type = "flash"
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = 1.0 / math.sqrt(self.head_dim)
        else:
            # Fallback to standard MultiheadAttention
            self.attention_type = "standard"
            self.self_attention = nn.MultiheadAttention(
                dim, num_heads, dropout=dropout_attn, batch_first=True
            )

        self.ln_2 = nn.LayerNorm(dim)

        # Optimized FFNN with better activation
        self.FFNN = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),  # Faster approximation
            nn.Dropout(dropout_FFNN),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_FFNN),
        )

    def _flash_attention_forward(self, x, mask, attn_mask=None):
        """Flash Attention implementation."""
        B, L, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply Flash Attention
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0 if not self.training else 0.1,
            scale=self.scale,
            is_causal=False,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(attn_output)

    def _standard_attention_forward(self, x, mask, attn_mask=None):
        """Standard attention fallback."""
        return self.self_attention(
            x,
            x,
            x,
            key_padding_mask=mask,
            attn_mask=attn_mask,
            need_weights=False,
            average_attn_weights=False,
        )[0]

    def _forward_impl(self, x, mask, attn_mask=None):
        """Core forward implementation."""
        z = self.ln_1(x)

        if self.attention_type == "flash":
            attn_output = self._flash_attention_forward(z, mask, attn_mask)
        else:
            attn_output = self._standard_attention_forward(z, mask, attn_mask)

        x = x + attn_output
        y = self.ln_2(x)
        x = x + self.FFNN(y)
        return x

    def forward(self, x, mask, attn_mask=None):
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, attn_mask
            )
        else:
            return self._forward_impl(x, mask, attn_mask)


class CachedMaskProcessor(nn.Module):
    """Advanced mask processor with pre-computation and caching capabilities."""

    def __init__(self, cache_size: int = 1000, precompute_masks: bool = True):
        super().__init__()
        self.cache_size = cache_size
        self.precompute_masks = precompute_masks
        self.mask_cache = {}
        self.position_cache = {}

    def _get_cache_key(self, x: Tensor, mask_type: str) -> str:
        """Generate cache key for mask computation."""
        # Use hash of tensor data and shape for caching
        if mask_type in ["causality", "euclidean"]:
            # For position-based masks, use position data
            pos_data = x[:, :, :3] if x.shape[-1] >= 3 else x
            return f"{mask_type}_{pos_data.shape}_{pos_data.data_ptr()}"
        else:
            # For ID masks, use unique values
            unique_vals = torch.unique(x)
            return f"{mask_type}_{x.shape}_{unique_vals.shape}_{unique_vals.data_ptr()}"

    def forward(self, x: Tensor, mask_module: nn.Module, mask_type: str) -> Tensor:
        """Forward pass with caching."""
        cache_key = self._get_cache_key(x, mask_type)

        # Check cache first
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        # Compute mask
        mask = mask_module(x)

        # Cache if within size limit
        if len(self.mask_cache) < self.cache_size:
            self.mask_cache[cache_key] = mask.detach().clone()

        return mask

    def clear_cache(self):
        """Clear the mask cache."""
        self.mask_cache.clear()
        self.position_cache.clear()


class OptimizedPairwiseProcessing(nn.Module):
    """More efficient pairwise processing with better memory usage."""

    def __init__(
        self,
        num_masks: int,
        dims: Union[List, int],
        num_heads: int,
        use_depthwise_separable: bool = True,  # Memory optimization
    ):
        super().__init__()

        if isinstance(dims, int):
            dims = [dims]

        if use_depthwise_separable and num_masks > 1:
            # Use depthwise separable convolutions for efficiency
            self.dw_conv = nn.Conv2d(
                in_channels=num_masks,
                out_channels=num_masks,
                kernel_size=1,
                groups=num_masks,  # Depthwise
                bias=False,
            )
            self.pw_conv = nn.Conv2d(
                in_channels=num_masks, out_channels=dims[0], kernel_size=1, bias=True
            )
            self.use_depthwise = True
        else:
            self.use_depthwise = False
            module_list = []
            current_channels = num_masks

            for dim in dims:
                module_list.extend(
                    [
                        nn.Conv2d(
                            in_channels=current_channels,
                            out_channels=dim,
                            kernel_size=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(dim),
                        nn.GELU(approximate="tanh"),
                    ]
                )
                current_channels = dim

            self.emb = nn.Sequential(*module_list)
            current_channels = dims[-1]

        # Final projection to num_heads
        final_channels = dims[-1] if not use_depthwise_separable else dims[0]
        self.final_proj = nn.Sequential(
            nn.Conv2d(
                in_channels=final_channels,
                out_channels=num_heads,
                kernel_size=1,
            ),
            nn.BatchNorm2d(num_heads),
            nn.GELU(approximate="tanh"),
        )

    def forward(self, x):
        if self.use_depthwise:
            # Depthwise separable convolution path
            x = self.dw_conv(x)
            x = self.pw_conv(x)
            x = self.final_proj(x)
        else:
            # Standard convolution path
            x = self.emb(x)
            x = self.final_proj(x)
        return x


class nuT_advanced_optimized(GNN):
    """
    Advanced optimized version of nuT model with comprehensive improvements:
    4. Architecture optimizations (Flash Attention, depthwise separable convolutions)
    5. I/O optimizations (caching, pre-computation, torch.compile)
    6. Memory management (gradient checkpointing, mixed precision support)
    7. Batch processing optimizations
    8. Bug fixes and code deduplication
    """

    def __init__(
        self,
        idx_dict: Dict,
        emb_dims: Union[List, int],
        seq_length: Optional[int],
        emb_type: str = "nuT",
        n_features: int = 8,
        abs_position_encoding: bool = True,
        refractive_index: Optional[float] = 1.33,
        masks: Optional[Union[List, str]] = [
            "Causality",
            "Euclidean",
            "DUs",
            "DOMs",
            "PMTs",
        ],
        mode: Optional[str] = "concat",
        pairwise_dims: Union[List, int] = 64,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_hits_blocks: int = 8,
        no_evt_blocks: Optional[int] = 4,
        # Advanced optimization parameters
        use_flash_attention: bool = True,
        use_gradient_checkpointing: bool = False,
        use_mixed_precision: bool = True,
        mask_cache_size: int = 1000,
        precompute_static_masks: bool = True,
        use_depthwise_separable: bool = True,
        enable_torch_compile: bool = True,
    ):
        """Construct advanced optimized Transformer."""
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__(n_features, model_dim)

        self.idx_dict = idx_dict
        self.seq_length = seq_length or 300
        self.n_features = n_features
        self.num_heads = num_heads
        self.mode = mode
        self.masks = [masks] if isinstance(masks, str) else masks
        self.use_flash_attention = use_flash_attention
        self.use_mixed_precision = use_mixed_precision
        self.enable_torch_compile = enable_torch_compile

        # Class token with better initialization
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)

        # Use optimized processing
        self.processing = self._create_optimized_processing(
            emb_type, model_dim, n_features
        )
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, self.seq_length)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        # Advanced mask processor with caching
        self.mask_processor = CachedMaskProcessor(
            cache_size=mask_cache_size, precompute_masks=precompute_static_masks
        )

        # Optimized pairwise mask modules
        self._setup_mask_modules(refractive_index, precompute_static_masks)

        # Optimized pairwise processing
        self.pw_processing = OptimizedPairwiseProcessing(
            len(self.masks or []),
            pairwise_dims,
            num_heads,
            use_depthwise_separable=use_depthwise_separable,
        )

        # Advanced encoder blocks
        self.hits_blocks = self._create_encoder_blocks(
            no_hits_blocks,
            model_dim,
            num_heads,
            dropout_attn,
            hidden_dim,
            dropout_FFNN,
            use_gradient_checkpointing,
        )

        self.evt_blocks = self._create_encoder_blocks(
            no_evt_blocks or 0,
            model_dim,
            num_heads,
            dropout_attn,
            hidden_dim,
            dropout_FFNN,
            use_gradient_checkpointing,
        )

    def _create_optimized_processing(
        self, emb_type: str, model_dim: int, n_features: int
    ):
        """Create optimized feature processing with bug fixes."""
        if emb_type == "nuT":
            return nn.Linear(n_features, model_dim)
        elif emb_type == "Kaggle":
            # Fixed bug: proper handling of emb_dims
            return nn.Sequential(
                nn.Linear(n_features, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(approximate="tanh"),
                nn.Linear(model_dim // 2, model_dim),
            )
        elif emb_type == "ParticleTransformer":
            return nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, model_dim),
                nn.GELU(approximate="tanh"),
            )
        else:
            raise ValueError(f"Unknown embedding type: {emb_type}")

    def _setup_mask_modules(
        self, refractive_index: Optional[float], cache_static: bool
    ):
        """Setup mask modules with proper type handling."""
        if self.masks:
            self.pw_causality = (
                OptimizedCausalityMask(refractive_index or 1.33)
                if "Causality" in self.masks
                else None
            )
            self.pw_euclidean = (
                OptimizedEuclideanMask(50) if "Euclidean" in self.masks else None
            )
            self.pw_du_ids = (
                OptimizedIdsMask(cache_static) if "DUs" in self.masks else None
            )
            self.pw_dom_ids = (
                OptimizedIdsMask(cache_static) if "DOMs" in self.masks else None
            )
            self.pw_pmt_ids = (
                OptimizedIdsMask(cache_static) if "PMTs" in self.masks else None
            )
        else:
            # Initialize all as None if no masks
            self.pw_causality = None
            self.pw_euclidean = None
            self.pw_du_ids = None
            self.pw_dom_ids = None
            self.pw_pmt_ids = None

    def _create_encoder_blocks(
        self,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        dropout_attn: float,
        hidden_dim: int,
        dropout_FFNN: float,
        use_checkpoint: bool,
    ) -> nn.ModuleList:
        """Create optimized encoder blocks."""
        return nn.ModuleList(
            [
                FlashAttentionEncoder(
                    dim=model_dim,
                    num_heads=num_heads,
                    dropout_attn=dropout_attn,
                    hidden_dim=hidden_dim,
                    dropout_FFNN=dropout_FFNN,
                    use_flash_attention=self.use_flash_attention,
                    use_checkpoint=use_checkpoint,
                )
                for _ in range(num_blocks)
            ]
        )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """cls_token should not be subject to weight decay during training."""
        return {"cls_token"}

    def _compute_masks_advanced(self, x0: Tensor, B: int, L: int) -> Optional[Tensor]:
        """Advanced mask computation with caching and vectorization."""
        if not self.masks or not self.mode:
            return None

        masks = []

        # Pre-extract all position data once
        position_data = None
        if self.pw_causality or self.pw_euclidean:
            x_pos = x0[:, :, self.idx_dict["pos_x"]].unsqueeze(-1)
            y_pos = x0[:, :, self.idx_dict["pos_y"]].unsqueeze(-1)
            z_pos = x0[:, :, self.idx_dict["pos_z"]].unsqueeze(-1)
            position_data = torch.cat((x_pos, y_pos, z_pos), dim=2)

        # Process masks with caching
        if self.pw_causality and position_data is not None:
            t = x0[:, :, self.idx_dict["t"]].unsqueeze(-1)
            spacetime_data = torch.cat((position_data, t), dim=2)
            mask_causality = self.mask_processor(
                spacetime_data, self.pw_causality, "causality"
            ).unsqueeze(1)
            masks.append(mask_causality)

        if self.pw_euclidean and position_data is not None:
            mask_euclidean = self.mask_processor(
                position_data, self.pw_euclidean, "euclidean"
            ).unsqueeze(1)
            masks.append(mask_euclidean)

        # Process ID masks efficiently
        id_configs = [
            (self.pw_du_ids, "du_id", "du"),
            (self.pw_dom_ids, "dom_id", "dom"),
            (self.pw_pmt_ids, "channel_id", "pmt"),
        ]

        for mask_module, id_key, cache_key in id_configs:
            if mask_module is not None:
                ids = x0[:, :, self.idx_dict[id_key]]
                mask = self.mask_processor(ids, mask_module, cache_key).unsqueeze(1)
                masks.append(mask)

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
        """Advanced optimized forward pass."""
        # Use autocast for mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._forward_impl(data)
        else:
            return self._forward_impl(data)

    def _forward_impl(self, data: Data) -> Tensor:
        """Core forward implementation."""
        x0, mask0, evt_length = array_to_sequence(data.x, data.batch, padding_value=0)
        B, L, _ = x0.shape

        # Class token creation (optimized)
        cls_token = self.cls_token.expand(B, -1, -1)  # More efficient than repeat

        # Feature filtering (vectorized)
        to_remove = {"trig", "du_id", "dom_id", "channel_id"}
        valid_indices = [
            idx for key, idx in self.idx_dict.items() if key not in to_remove
        ]
        x = x0[:, :, valid_indices]

        # Features processing and position encoding
        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)

        # Advanced mask computation
        attn_mask = self._compute_masks_advanced(x0, B, L)

        # Optimized padding mask creation
        mask_dtype = attn_mask.dtype if attn_mask is not None else torch.float32
        mask = torch.zeros(mask0.shape, dtype=mask_dtype, device=mask0.device)
        mask[~mask0] = -torch.inf

        # Process through blocks with optimized flow
        if (self.no_evt_blocks is None) or (self.no_evt_blocks == 0):
            # Single-pass mode
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (1, 0, 1, 0))

            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask=attn_mask)
        else:
            # Two-pass mode
            # First pass: hits only
            for hits_block in self.hits_blocks:
                x = hits_block(x, mask=mask, attn_mask=attn_mask)

            # Second pass: with event token
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            for evt_block in self.evt_blocks:
                x = evt_block(x, mask=mask)

        return x[:, 0]

    def clear_caches(self):
        """Clear all caches to free memory."""
        self.mask_processor.clear_cache()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and optimization info."""
        return {
            "model_type": "nuT_advanced_optimized",
            "flash_attention": self.use_flash_attention,
            "mixed_precision": self.use_mixed_precision,
            "gradient_checkpointing": any(
                block.use_checkpoint for block in self.hits_blocks
            ),
            "mask_cache_size": len(self.mask_processor.mask_cache),
            "sequence_length": self.seq_length,
            "num_heads": self.num_heads,
            "model_dim": self.out_dim,
        }


# Optional: Enable torch.compile for additional optimization
try:
    if torch.cuda.is_available() and torch.__version__ >= "2.0":
        nuT_advanced_optimized = torch.compile(
            nuT_advanced_optimized, mode="reduce-overhead", fullgraph=True
        )
except Exception:
    pass  # torch.compile not available or failed
