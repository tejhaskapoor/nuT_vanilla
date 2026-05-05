"""Optimized version of nuT model with performance improvements — GraphNet-free.

All GraphNet and torch_geometric dependencies have been removed. The model
inherits from ``nn.Module`` and uses a local pure-PyTorch implementation of
``array_to_sequence``.

Optimization techniques used (documented at each site):
  - torch.cdist for vectorised pairwise distances (O(BL²) with one CUDA kernel)
  - Numerical-stability epsilon in spacetime-interval square-root
  - In-place clamp_ to avoid extra tensor allocations
  - ID-mask caching for static detector geometry
  - need_weights=False in MHA to skip materialising the attention-weight matrix
  - GELU(approximate="tanh") polynomial approximation (faster than exact erf)
  - Gradient checkpointing to trade recomputation for peak memory
  - Position column pre-extraction to avoid repeated fancy-indexing in mask loop
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Union

from torch import Tensor

from .model_components import (
    FeaturesProcessing,
    AbsolutePositionalEncoding,
    PairwiseProcessing,
    #CausalityMask,
    #EuclideanMask,
    #IdsMask,
    #Encoder_block,
)

from .data_representation import array_to_sequence


# ---------------------------------------------------------------------------
# Detector-specific configuration (mirrors nuT_model_no_graphnet.py)
# ---------------------------------------------------------------------------
_DETECTOR_CONFIGS = {
    "KM3NeT": {
        "id_cols_to_remove": ["trig", "du_id", "dom_id", "channel_id"],
        "pos_keys": ("pos_x", "pos_y", "pos_z"),
        "default_masks": ["Causality", "Euclidean", "DUs", "DOMs", "PMTs"],
    },
    "Prometheus": {
        "id_cols_to_remove": ["is_signal", "string_id"],
        "pos_keys": ("sensor_pos_x", "sensor_pos_y", "sensor_pos_z"),
        "default_masks": ["Causality", "Euclidean", "STRING"],
    },
}


# ---------------------------------------------------------------------------
# Optimized mask modules
# ---------------------------------------------------------------------------

class OptimizedCausalityMask(nn.Module):
    """Causality (Cherenkov-cone) pairwise mask.

    Computes the signed spacetime interval between every pair of hits.

    Optimization: uses ``torch.cdist`` for the spatial L2 distance, which
    dispatches to a single fused CUDA kernel (batched matrix multiply) and
    avoids explicitly broadcasting O(BL²·3) intermediate tensors.
    """

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
        """x: [B, L, 4]  (x, y, z, t)."""
        pos = x[:, :, :3] * self.scaling_xyz
        time = x[:, :, 3] * self.scaling_t

        # torch.cdist computes all pairwise L2 distances in one call — much
        # faster than manual subtraction + norm, especially on GPU.
        pos_diff = torch.cdist(pos, pos, p=2)           # [B, L, L]
        time_diff = (time.unsqueeze(2) - time.unsqueeze(1)) * self.v

        spacetime_interval = pos_diff.pow(2) - time_diff.pow(2)

        # +1e-8 epsilon: prevents NaN gradient from sqrt(0) at identical hits.
        four_distance = torch.sign(spacetime_interval) * torch.sqrt(
            torch.abs(spacetime_interval) + 1e-8
        )

        # clamp_ is an in-place operation: it modifies the tensor directly
        # instead of allocating a new one, saving one memory round-trip.
        return four_distance.clamp_(-4, 4)


class OptimizedEuclideanMask(nn.Module):
    """Euclidean inter-hit distance pairwise mask.

    Optimization: uses ``torch.cdist`` for a single-kernel batched distance
    computation; in-place ``clamp_`` avoids a temporary allocation.
    """

    def __init__(self, max_distance: float = 50.0):
        super().__init__()
        self.max_distance = max_distance

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L, 3]  (x, y, z positions)."""
        # torch.cdist dispatches to cuBLAS under the hood on GPU — much faster
        # than expanding tensors and calling torch.norm manually.
        euclidean_distance = torch.cdist(x, x, p=2)    # [B, L, L]
        return euclidean_distance.clamp_(0, self.max_distance)  # in-place


class OptimizedIdsMask(nn.Module):
    """Binary same-ID pairwise mask with optional caching.

    Optimization: vectorised broadcasting comparison (no Python loop); optional
    caching skips recomputation entirely when the detector geometry is static
    (same string/DOM/PMT IDs across batches, e.g. fixed Monte-Carlo geometry).
    """

    def __init__(self, cache_static: bool = True):
        super().__init__()
        self.cache_static = cache_static
        # Cache state — valid only when the ID tensor is unchanged from the
        # previous call.  We store the IDs to detect changes via torch.equal.
        self._cached_mask: Optional[Tensor] = None
        self._cached_ids: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, L]  (integer IDs for each hit).

        Returns: [B, L, L] float mask; 1 where IDs match, 0 elsewhere.
        """
        # Cache hit: if the ID tensor is identical to last call, reuse result.
        if self.cache_static and self._cached_mask is not None:
            if self._cached_ids.device == x.device and torch.equal(x, self._cached_ids):
                return self._cached_mask

        # Vectorised outer equality — no Python loop, single CUDA kernel.
        mask = (x.unsqueeze(2) == x.unsqueeze(1)).float()   # [B, L, L]

        if self.cache_static:
            self._cached_mask = mask
            self._cached_ids = x.clone()

        return mask


# ---------------------------------------------------------------------------
# Optimized encoder block
# ---------------------------------------------------------------------------

class OptimizedEncoder_block(nn.Module):
    """Transformer encoder block with several performance improvements.

    Optimizations:
    - ``need_weights=False``: tells PyTorch's MHA not to compute or return the
      attention weight matrix (the B×H×L×L softmax output).  This saves a
      significant amount of memory and compute when weights are not inspected.
    - ``nn.GELU(approximate="tanh")``: uses a polynomial (tanh-based)
      approximation of GELU instead of the exact erf integral — roughly 2×
      faster on typical hardware with negligible accuracy loss.
    - Gradient checkpointing (``use_checkpoint=True``): during the backward
      pass PyTorch recomputes the forward activations instead of storing them,
      trading extra FLOPs for lower peak GPU memory.  Useful for very deep
      models or long sequences.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout_attn, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(dim)

        # GELU(approximate="tanh"): polynomial approximation via tanh, avoids
        # the exact erf call which is slower on most hardware backends.
        self.FFNN = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_FFNN),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_FFNN),
        )

    def _forward_impl(self, x: Tensor, mask: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Core forward logic; split out so it can be wrapped by checkpoint."""
        z = self.ln_1(x)
        # need_weights=False: skip computing the attention-weight matrix.
        # PyTorch's fused FlashAttention kernel is also only used when
        # need_weights=False, so this can give additional speedup on A100/H100.
        attn_output = self.self_attention(
            z, z, z,
            key_padding_mask=mask,
            attn_mask=attn_mask,
            need_weights=False,
            average_attn_weights=False,
        )[0]
        x = x + attn_output
        x = x + self.FFNN(self.ln_2(x))
        return x

    def forward(self, x: Tensor, mask: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        if self.use_checkpoint and self.training:
            # Gradient checkpointing: recompute activations on the backward
            # pass instead of storing them — reduces peak memory at the cost
            # of ~33 % extra forward FLOPs.
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, mask, attn_mask
            )
        return self._forward_impl(x, mask, attn_mask)


# ---------------------------------------------------------------------------
# Optimized feature embedding
# ---------------------------------------------------------------------------

class OptimizedFeaturesProcessing(nn.Module):
    """Feature projection (hit features → model dimension).

    Supports three embedding strategies matching the base nuT class.

    Optimizations vs. the original:
    - Bug fix: ``emb_dims[0]`` (was ``emb_dim[0]``) in Kaggle branch.
    - GELU(approximate="tanh") throughout for faster non-linearity.
    """

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
            self.emb = nn.Sequential(
                nn.Linear(n_features, emb_dims[0]),   # Bug fix: was emb_dim[0]
                nn.LayerNorm(emb_dims[0]),
                nn.GELU(approximate="tanh"),           # faster approx
                nn.Linear(emb_dims[0], emb_dims[1]),
            )

        elif emb_type == "ParticleTransformer":
            if isinstance(emb_dims, int):
                emb_dims = [emb_dims]
            self.model_dim = emb_dims[-1]
            module_list = []
            for emb_dim in emb_dims:
                module_list.extend([
                    nn.LayerNorm(n_features),
                    nn.Linear(n_features, emb_dim),
                    nn.GELU(approximate="tanh"),       # faster approx
                ])
                n_features = emb_dim
            self.emb = nn.Sequential(*module_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x) * math.sqrt(self.model_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class nuT_optimized(nn.Module):
    """Optimized nuT transformer — pure PyTorch, no GraphNet dependency.

    Identical in architecture to ``nuT`` but uses the optimized sub-modules
    above and adds gradient-checkpointing and static-mask-caching options.

    Key optimizations (see individual classes for details):
    1. ``torch.cdist`` for pairwise distance masks.
    2. In-place ``clamp_`` to avoid extra allocations.
    3. ID-mask caching for static detector geometry.
    4. ``need_weights=False`` in MHA.
    5. ``GELU(approximate="tanh")`` in feed-forward networks.
    6. Optional gradient checkpointing per encoder block.
    7. Position columns extracted once before the mask loop.
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
        use_gradient_checkpointing: bool = False,
        cache_static_masks: bool = True,
        detector_type: str = "Prometheus",
    ):
        """Construct optimized nuT transformer.

        Args:
            use_gradient_checkpointing: Enables gradient checkpointing inside
                each encoder block — lowers peak memory at the cost of ~33 %
                extra forward compute during training.
            cache_static_masks: Caches ID-based pairwise masks when the
                detector geometry is fixed across batches (Prometheus default).
            detector_type: ``"KM3NeT"`` or ``"Prometheus"`` — controls which
                columns are stripped as metadata before the transformer.
        """
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__()

        # nb_inputs / nb_outputs are queried by the training wrapper
        self.nb_inputs = n_features
        self.nb_outputs = model_dim

        if detector_type not in _DETECTOR_CONFIGS:
            raise ValueError(
                f"Unknown detector_type '{detector_type}'. "
                f"Choose from {list(_DETECTOR_CONFIGS.keys())}."
            )
        det_cfg = _DETECTOR_CONFIGS[detector_type]
        # Metadata columns stripped before passing hits to the transformer
        self._id_cols_to_remove: List[str] = det_cfg["id_cols_to_remove"]
        # Position-coordinate key names used in the mask computation
        self._pos_keys: Tuple[str, str, str] = det_cfg["pos_keys"]

        self.idx_dict = idx_dict
        self.seq_length = seq_length
        self.n_features = n_features
        self.num_heads = num_heads
        self.mode = mode
        self.masks = [masks] if isinstance(masks, str) else masks
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.cache_static_masks = cache_static_masks

        # Learnable CLS token aggregates global event information
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        self.processing = OptimizedFeaturesProcessing(model_dim, emb_type, n_features)
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, seq_length or 300)

        self.no_hits_blocks = no_hits_blocks
        self.no_evt_blocks = no_evt_blocks

        # Instantiate only the masks that are requested
        self.pw_causality = (
            OptimizedCausalityMask(refractive_index or 1.33)
            if self.masks and "Causality" in self.masks else None
        )
        self.pw_euclidean = (
            OptimizedEuclideanMask(50)
            if self.masks and "Euclidean" in self.masks else None
        )
        self.pw_string_id = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "STRING" in self.masks else None
        )
        self.pw_du_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "DUs" in self.masks else None
        )
        self.pw_dom_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "DOMs" in self.masks else None
        )
        self.pw_pmt_ids = (
            OptimizedIdsMask(cache_static_masks)
            if self.masks and "PMTs" in self.masks else None
        )

        if mode == "concat":
            self.pw_processing = PairwiseProcessing(
                len(self.masks or []), pairwise_dims, num_heads
            )
        elif mode == "sum":
            self.pw_processing = PairwiseProcessing(1, pairwise_dims, num_heads)

        # Encoder blocks — all use the optimized variant
        def _make_blocks(n: int) -> nn.ModuleList:
            return nn.ModuleList([
                OptimizedEncoder_block(
                    model_dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN,
                    use_checkpoint=use_gradient_checkpointing,
                )
                for _ in range(n)
            ])

        self.hits_blocks = _make_blocks(no_hits_blocks)
        self.evt_blocks = _make_blocks(no_evt_blocks)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """CLS token should not be subject to weight decay during training."""
        return {"cls_token"}

    def _compute_masks_optimized(self, x0: Tensor, B: int, L: int) -> Optional[Tensor]:
        """Build and combine pairwise attention masks.

        Optimization: position columns (x, y, z) are extracted **once** before
        the mask loop, avoiding repeated fancy-indexing on large [B, L, F]
        tensors inside each mask's forward call.
        """
        if not self.masks or not self.mode:
            return None

        masks = []
        px_key, py_key, pz_key = self._pos_keys

        if self.pw_causality or self.pw_euclidean:
            # Pre-extract 3D positions once — reused by both causality and
            # Euclidean masks rather than re-indexing x0 twice.
            positions = torch.cat([
                x0[:, :, self.idx_dict[px_key]].unsqueeze(-1),
                x0[:, :, self.idx_dict[py_key]].unsqueeze(-1),
                x0[:, :, self.idx_dict[pz_key]].unsqueeze(-1),
            ], dim=2)                                       # [B, L, 3]

            if self.pw_causality:
                t = x0[:, :, self.idx_dict["t"]].unsqueeze(-1)
                spacetime_data = torch.cat((positions, t), dim=2)  # [B, L, 4]
                masks.append(self.pw_causality(spacetime_data).unsqueeze(1))

            if self.pw_euclidean:
                masks.append(self.pw_euclidean(positions).unsqueeze(1))

        # ID-based masks
        if self.pw_string_id:
            masks.append(
                self.pw_string_id(x0[:, :, self.idx_dict["string_id"]]).unsqueeze(1)
            )
        if self.pw_du_ids:
            masks.append(
                self.pw_du_ids(x0[:, :, self.idx_dict["du_id"]]).unsqueeze(1)
            )
        if self.pw_dom_ids:
            masks.append(
                self.pw_dom_ids(x0[:, :, self.idx_dict["dom_id"]]).unsqueeze(1)
            )
        if self.pw_pmt_ids:
            masks.append(
                self.pw_pmt_ids(x0[:, :, self.idx_dict["channel_id"]]).unsqueeze(1)
            )

        if not masks:
            return None

        # Stack all mask channels: [B, n_masks, L, L]
        masks_tensor = torch.cat(masks, dim=1)

        if self.mode == "sum":
            # Collapse channels before projection
            masks_tensor = torch.sum(masks_tensor, dim=1, keepdim=True)

        # Project n_masks channels → num_heads bias values: [B*num_heads, L, L]
        return self.pw_processing(masks_tensor).view(B * self.num_heads, L, L)

    def forward(self, data) -> Tensor:
        """Forward pass: hits → event embedding (CLS token).

        Args:
            data: Dict with keys ``"x"`` ([N, d]) and ``"batch"`` ([N]), or a
                PyG ``Data`` object — both are supported.

        Returns:
            Event embedding of shape ``[B, model_dim]``.
        """
        # Accept both plain dicts and PyG Data objects (GraphNet-free interface)
        _x = data["x"] if isinstance(data, dict) else data.x
        _batch = data["batch"] if isinstance(data, dict) else data.batch

        # Convert flat [N, d] to padded [B, L, d] using local implementation
        x0, mask0, evt_length = array_to_sequence(_x, _batch, padding_value=0)
        B, L, _ = x0.shape

        cls_token = self.cls_token.repeat(B, 1, 1)

        # Strip metadata columns (trigger flags, IDs) — used only for pairwise
        # masks; they should not enter the transformer as physics features.
        filtered_cols = [
            v for k, v in self.idx_dict.items()
            if k not in self._id_cols_to_remove
        ]
        x = x0[:, :, filtered_cols]

        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)

        attn_mask = self._compute_masks_optimized(x0, B, L)

        # Padding mask: -inf for padded positions so attention ignores them
        pad_mask = torch.zeros(
            mask0.shape,
            dtype=attn_mask.dtype if attn_mask is not None else torch.float32,
            device=mask0.device,
        )
        pad_mask[~mask0] = -torch.inf

        if (self.no_evt_blocks is None) or (self.no_evt_blocks == 0):
            # No separate event blocks: CLS token present from the first block
            x = torch.cat([cls_token, x], dim=1)
            cls_pad = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (1, 0, 1, 0))
            for block in self.hits_blocks:
                x = block(x, mask=pad_mask, attn_mask=attn_mask)
        else:
            # Hit blocks first (no CLS — local hit interactions only)
            for block in self.hits_blocks:
                x = block(x, mask=pad_mask, attn_mask=attn_mask)

            # Event blocks: prepend CLS, run global attention (no pairwise mask)
            x = torch.cat([cls_token, x], dim=1)
            cls_pad = torch.zeros((B, 1), dtype=x0.dtype, device=mask0.device)
            pad_mask = torch.cat([cls_pad, pad_mask], dim=1)
            for block in self.evt_blocks:
                x = block(x, mask=pad_mask)

        # Return the CLS token as the event-level embedding
        return x[:, 0]


# Optional: torch.compile for additional optimization (PyTorch 2.0+)
# try:
#     nuT_optimized = torch.compile(nuT_optimized, mode="reduce-overhead")
# except AttributeError:
#     pass  # torch.compile not available in this PyTorch version

