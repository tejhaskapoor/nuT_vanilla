"""Embedding, positional encoding, pairwise masks, and encoder blocks for nuT.

Classes:
    FeaturesProcessing       — projects raw hit features into model dimension
    AbsolutePositionalEncoding — sinusoidal position encoding
    PairwiseProcessing       — Conv2d projection of stacked mask channels
    Encoder_block            — transformer encoder block (MHA + FFNN)
"""


import torch
import torch.nn as nn
from torch.nn.functional import linear
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.functional import Tensor
import math


from typing import Any, Callable, Optional, Sequence, Union, List


class FeaturesProcessing(nn.Module):
    """ Process the hits features by passing them through a embedding block. """

    def __init__(
                    self,
                    emb_type: str = "nuT",
                    emb_dims: Union[List, int] = 128,
                    n_features: int = 6,
    ):
        """ Pass all the features through a embedding block before feed them to the model.

            Args:
                emb_type: The type of embedding used ("nuT", "Kaggle", "ParticleTransformer"). Default: "nuT"
                n_features: The number of features in the input data.
                emb_dims: Dimensionality of the consecutive linear layers.
        """

        super().__init__()

        if emb_type == "nuT":
            assert isinstance(emb_dims, int), f"Only one embedding dimension is possible while {emb_dims} was provided"
            self.emb = nn.Linear(n_features, emb_dims)
            self.model_dim = emb_dims
        elif emb_type == "Kaggle":
            assert len(emb_dims) == 2, f"Only one two embedding dimension are possible while {emb_dims} was provided"
            self.model_dim = emb_dims[-1]
            module_list = []
            module_list.extend([
                nn.Linear(n_features, emb_dims[0]),
                nn.LayerNorm(emb_dims[0]),
                nn.GELU(),
                nn.Linear(emb_dims[0], emb_dims[1]),
            ])
            self.emb = nn.Sequential(*module_list)
        elif emb_type == "ParticleTransformer":
            if isinstance(emb_dims, int):
                emb_dims = [emb_dims]
            self.model_dim = emb_dims[-1]
            module_list = []
            for emb_dim in emb_dims:
                module_list.extend([
                                        nn.LayerNorm(n_features),
                                        nn.Linear(n_features, emb_dim),
                                        nn.GELU()
                ])
                n_features = emb_dim
            self.emb = nn.Sequential(*module_list)

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.model_dim)


class AbsolutePositionalEncoding(nn.Module):
    """Absolute sinusoidal position encoding for sequences."""

    def __init__(
                    self,
                    dim: int = 128,
                    seq_length: int = 300,
    ):
        """ Associate an unique representation to each position in a sequence using Sinusoidal Fourier position encoding.

        Args:
            dim: Dimension of the model
            seq_length: Maximun length of the sequence.

    """

        super().__init__()

        pos_emb = torch.zeros(seq_length, dim)
        positions = torch.arange(0, seq_length, dtype = torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, dim, 2).float() * -math.log(10000.0) / dim  )

        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term)

        pos_emb = pos_emb.unsqueeze(0)  # pos_emb.shape: [1, seq_length, dim]

        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x):
        # x.shape = [B, seq_len, dim]
        x = x + self.pos_emb[:, :x.shape[1], :]
        return x
    
class PairwiseProcessing(nn.Module):
    """Projects stacked pairwise mask channels into per-head attention biases.

    Takes an input tensor of shape ``[B, num_masks, L, L]`` (where each channel
    is one pairwise mask, e.g. causality, Euclidean distance, ID match) and
    outputs ``[B, num_heads, L, L]`` attention biases via a stack of 1×1 Conv2d
    layers with BatchNorm and GELU activations.
    """

    def __init__(
        self,
        num_masks: int,
        dims: Union[List, int],
        num_heads: int,
    ):
        """Construct PairwiseProcessing.

        Args:
            num_masks: Number of input mask channels (``C_in``).
            dims: Hidden channel size(s) for intermediate Conv2d layers.
                Pass an int for a single hidden layer, or a list for multiple.
            num_heads: Number of output channels (= number of attention heads).
        """

        super().__init__()

        if isinstance(dims, int):
            dims = [dims]

        module_list = []
        for dim in dims:
            module_list.extend([
                nn.Conv2d(
                    in_channels = num_masks, 
                    out_channels = dim, 
                    kernel_size = 1
                ),
                nn.BatchNorm2d(dim),
                nn.GELU()
            ])
            num_masks = dim
            
        module_list.extend([
            nn.Conv2d(
                in_channels = dims[-1], 
                out_channels = num_heads, 
                kernel_size = 1
            ),
            nn.BatchNorm2d(num_heads),
            nn.GELU()
        ])
        
        self.emb = nn.Sequential(*module_list)

    def forward(self, x):
        return self.emb(x)


class Encoder_block(nn.Module):
    """ Encoder block for Transformer model. """
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        dropout_attn: float = 0.2,
        hidden_dim: int = 256,
        dropout_FFNN: float = 0.2,
    ):
        """
            Input data goes through encoder block with MHA and a FFNN with GELU activation function.

            Args:
                dim: Dimension of the model.
                num_heads: Number of heads in MHA.
                dropout_attn: Dropout to be applied in MHA.
                hidden_dim: Dimension of FFNN.
                dropout_FFNN: Dropout to be applied in MHA.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim, 
            num_heads, 
            dropout = dropout_attn, 
            batch_first = True
        )
        self.ln_2 = nn.LayerNorm(dim)
        self.FFNN = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_FFNN),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_FFNN)
        )

    def forward(self, x, mask, attn_mask = None):
        z = self.ln_1(x)
        x = x + self.self_attention(
            z, z, z,
            key_padding_mask=mask,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,  # return per-head weights (not averaged)
        )[0]
        y = self.ln_2(x)
        x = x + self.FFNN(y)

        return x

