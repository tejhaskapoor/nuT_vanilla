"""Transformer models for neutrino reconstruction (nuT).

Classes
-------
nuT
    Transformer with physics-informed pairwise attention masks.
    Supports KM3NeT and Prometheus detectors via ``detector_type``.
    ``nuT_PROMETHEUS`` is a convenience alias.

nuT_vanilla
    Vanilla transformer — no pairwise masks, unrestricted attention
    across all hits (only padded positions are masked).
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
    CausalityMask,
    EuclideanMask,
    IdsMask,
    Encoder_block,
)

from .data_representation import array_to_sequence

# Detector-specific configuration: which columns to strip before the transformer
# (metadata used only for pairwise masks, not as physics inputs) and which
# position-coordinate key names to use.
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



class nuT_vanilla(nn.Module):
    """Vanilla transformer for neutrino telescopes — no pairwise masks.

    Processes a padded sequence of photon hits and outputs a single
    event-level embedding (the CLS token), shape ``[B, model_dim]``.

    Unlike ``nuT``, this model applies no physics-informed attention biases.
    All hits attend to all other hits freely; only padded positions are masked.
    """

    def __init__(
        self,
        n_features: int,
        emb_dims: Union[List, int],
        seq_length: Union[int, None],
        emb_type: str = "nuT",
        abs_position_encoding: bool = True,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
        no_blocks: int = 8,
    ):
        """Construct the vanilla nuT transformer.

        Args:
            n_features: Number of input features per hit (all columns are used
                directly — no metadata stripping).
            emb_dims: Embedding dimension(s). If a list, the last value is
                used as the model dimension.
            seq_length: Maximum sequence length (number of hits per event).
            emb_type: Embedding type for feature projection. One of
                ``"nuT"`` (linear), ``"Kaggle"`` (2-layer MLP),
                ``"ParticleTransformer"`` (multi-layer).
            abs_position_encoding: Whether to add sinusoidal position encoding.
            num_heads: Number of attention heads.
            dropout_attn: Dropout in multi-head attention.
            hidden_dim: Hidden dimension of the feed-forward network.
            dropout_FFNN: Dropout in the feed-forward network.
            no_blocks: Number of encoder blocks.
        """
        model_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[-1]
        super().__init__()

        # nb_inputs / nb_outputs are queried by the training wrapper
        self.nb_inputs = n_features
        self.nb_outputs = model_dim

        self.n_features = n_features
        self.num_heads = num_heads

        # Learnable CLS token: prepended to the hit sequence to aggregate
        # global event information
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Feature embedding and position encoding
        self.processing = FeaturesProcessing(emb_type, model_dim, n_features)
        self.abs_position_encoding = abs_position_encoding
        self.pos_enc = AbsolutePositionalEncoding(model_dim, seq_length)

        # Encoder blocks
        self.blocks = nn.Sequential(
            *[Encoder_block(model_dim, num_heads, dropout_attn, hidden_dim, dropout_FFNN)
              for _ in range(no_blocks)]
        )

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        """CLS token should not be subject to weight decay during training."""
        return {"cls_token"}

    def forward(self, data) -> Tensor:
        """Forward pass: hits → event embedding (CLS token).

        Args:
            data: Dict with keys ``"x"`` (flat hit tensor ``[N, d]``) and
                ``"batch"`` (event index per hit ``[N]``), or a PyG Data object.

        Returns:
            Event embedding tensor of shape ``[B, model_dim]``.
        """
        _x = data["x"] if isinstance(data, dict) else data.x
        _batch = data["batch"] if isinstance(data, dict) else data.batch

        # Convert flat [N, d] tensor to padded [B, L, d] sequence
        x, mask, _ = array_to_sequence(_x, _batch, padding_value=0)
        B, L, _ = x.shape

        # Embed hit features into model_dim
        x = self.processing(x)
        if self.abs_position_encoding:
            x = self.pos_enc(x)

        # Prepend CLS token: [B, 1+L, model_dim]
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        # Padding mask: 0 for real hits, -inf for padded positions.
        # CLS token is always unmasked (prepend a zero column).
        pad_mask = torch.zeros(B, L, dtype=x.dtype, device=x.device)
        pad_mask[~mask] = -torch.inf
        cls_pad = torch.zeros(B, 1, dtype=x.dtype, device=x.device)
        pad_mask = torch.cat([cls_pad, pad_mask], dim=1)

        # Run through encoder blocks (no pairwise attn_mask)
        for block in self.blocks:
            x = block(x, mask=pad_mask)

        # Return the CLS token as the event-level embedding
        return x[:, 0]
