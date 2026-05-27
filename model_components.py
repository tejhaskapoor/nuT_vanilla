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
import torch.nn.functional as F
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.functional import Tensor
import math


from typing import Any, Callable, Optional, Sequence, Union, List

if torch.cuda.is_available():
    CUDA_ARCH = torch.cuda.get_device_capability(0)[0]
    if CUDA_ARCH >= 9:  # Hopper/Blackwell: try FA4 first
        try:
            from flash_attn.cute import flash_attn_varlen_func
        except ImportError:
            from flash_attn import flash_attn_varlen_func
    elif CUDA_ARCH >= 8:  # Ampere: FA2
        from flash_attn import flash_attn_varlen_func


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
            need_weights=False,
        )[0]
        y = self.ln_2(x)
        x = x + self.FFNN(y)

        return x


def batch2offset(
    batch: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    seqlens = batch.unique(sorted=True, return_counts=True)[-1]
    return seqlens


class ConcatUnpaddedListTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, seqlens_list: list[torch.Tensor], *values_list: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = seqlens_list[0].device
        num_lists = len(seqlens_list)
        batch_size = seqlens_list[0].numel()
        
        stacked_seqlens = torch.stack(seqlens_list, dim=1) # (B, num_lists)
        final_seqlens = stacked_seqlens.sum(dim=1)
        
        flattened_seqlens = stacked_seqlens.reshape(-1)
        output_offsets = F.pad(torch.cumsum(flattened_seqlens, dim=0)[:-1], (1, 0)).reshape(batch_size, num_lists)
        
        source_offsets = F.pad(torch.cumsum(stacked_seqlens, dim=0)[:-1, :], (0, 0, 1, 0)) # (B, num_lists)

        indices_list = []
        for i in range(num_lists):
            base_dst = torch.repeat_interleave(output_offsets[:, i], seqlens_list[i])
            base_src_offset = torch.repeat_interleave(source_offsets[:, i], seqlens_list[i])
            
            num_tokens_i = values_list[i].shape[0]
            rel_pos = torch.arange(num_tokens_i, device=device) - base_src_offset
            
            dst_indices = base_dst + rel_pos
            indices_list.append(dst_indices)

        target_dtype = values_list[0].dtype
        output = torch.empty((final_seqlens.sum().item(), *values_list[0].shape[1:]), 
                             dtype=target_dtype, device=device)
        
        for i in range(num_lists):
            output[indices_list[i]] = values_list[i].to(target_dtype)

        ctx.mark_non_differentiable(final_seqlens)
        ctx.save_for_backward(*indices_list)
        return output, final_seqlens

    @staticmethod
    def backward(ctx, grad_output, grad_final_seqlens):
        indices_list = ctx.saved_tensors
        grad_values_list = [grad_output[indices] for indices in indices_list]
        return None, *grad_values_list


def concat_tokens_from_seqlens(
    tokens_list: list[torch.Tensor],
    seqlens_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_seqlens = ConcatUnpaddedListTensors.apply(seqlens_list, *tokens_list)
    return output, final_seqlens


class SplitUnpaddedTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, seqlens_list: list[torch.Tensor]) -> Sequence[torch.Tensor]:
        device = values.device
        num_lists = len(seqlens_list)
        batch_size = seqlens_list[0].numel()

        stacked_seqlens = torch.stack(seqlens_list, dim=1) # (B, num_lists)
        flattened_seqlens = stacked_seqlens.reshape(-1)
        
        source_offsets = F.pad(torch.cumsum(flattened_seqlens, dim=0)[:-1], (1, 0)).reshape(batch_size, num_lists)
        dst_offsets = F.pad(torch.cumsum(stacked_seqlens, dim=0)[:-1, :], (0, 0, 1, 0)) # (B, num_lists)

        outputs = []
        indices_list = []

        for i in range(num_lists):
            current_seqlens = seqlens_list[i]
            num_tokens_i = current_seqlens.sum().item()
            
            base_src = torch.repeat_interleave(source_offsets[:, i], current_seqlens)
            base_dst_offset = torch.repeat_interleave(dst_offsets[:, i], current_seqlens)
            
            rel_pos = torch.arange(num_tokens_i, device=device) - base_dst_offset
            
            src_indices = base_src + rel_pos
            indices_list.append(src_indices)
            outputs.append(values[src_indices])
        
        ctx.save_for_backward(*indices_list)
        ctx.total_tokens = values.shape[0]
        ctx.output_shape_suffix = values.shape[1:]
        
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        indices_list = ctx.saved_tensors
        device = indices_list[0].device
        
        grad_values = torch.zeros((ctx.total_tokens, *ctx.output_shape_suffix), 
                                  dtype=grad_outputs[0].dtype, device=device)
        
        for i, indices in enumerate(indices_list):
            grad_values[indices] = grad_outputs[i]
            
        return grad_values, None


def split_tokens_from_seqlens(
    tokens: torch.Tensor,
    seqlens_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    return SplitUnpaddedTensor.apply(tokens, seqlens_list)


# https://discuss.pytorch.org/t/index-of-first-occurrence-on-sorted-1d-tensor/113460/2
def arg_first_occurrence(x: torch.Tensor) -> torch.Tensor:
    n = x.numel()
    lbl = torch.arange(n, device=x.device)
    msk = x == lbl.unsqueeze(1)
    mna = (lbl + 1) * msk
    mnb = torch.where(mna != 0, mna, (n + 1) * x.new_ones(n, 1).long())
    foa = mnb.min(dim=1)[0] - 1
    fob = torch.where(foa != n, foa, -x.new_ones(n).long())
    return fob


def scatter_at_first_index(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None
) -> torch.Tensor:
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    first_occ = arg_first_occurrence(index)
    new_index = first_occ[first_occ != -1]
    out = torch.index_select(src, dim, new_index)
    return out


class AbsolutePositionalEncoding_varlen(nn.Module):
    """Sinusoidal positional encoding for flattened variable-length batches."""

    def __init__(self, dim: int = 128, seq_length: int = 300):
        super().__init__()

        pos_emb = torch.zeros(seq_length, dim)
        positions = torch.arange(seq_length, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dim)
        )

        pos_emb[:, 0::2] = torch.sin(positions * div_term)
        pos_emb[:, 1::2] = torch.cos(positions * div_term[: pos_emb[:, 1::2].shape[1]])

        self.register_buffer("pos_emb", pos_emb)  # [seq_length, dim]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: flattened batch, shape [S, dim]
            lengths: sequence lengths, shape [B], sum(lengths) == S

        Returns:
            x + positional encoding, shape [S, dim]
        """
        # positions = [0, 1, ..., L1-1, 0, 1, ..., L2-1, ...]
        positions = torch.arange(x.size(0), device=x.device)
        offsets = torch.repeat_interleave(torch.cumsum(lengths, dim=0) - lengths, lengths)
        positions = positions - offsets

        return x + self.pos_emb.index_select(0, positions)


class Attention_varlen(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(T, self.num_heads, self.head_dim)
        k = k.reshape(T, self.num_heads, self.head_dim)
        v = v.reshape(T, self.num_heads, self.head_dim)
        
        max_seqlen = torch.amax(seqlens).item()
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.torch.int32), (1, 0))
        x = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout,
            softmax_scale=self.scale,
            causal=False,
        ) # (T, nh, hd)
        x = x.reshape(T, C)
        
        x = self.proj(x)
        return x
    

class Encoder_block_varlen(nn.Module):
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
        self.self_attention = Attention_varlen(
            dim, 
            num_heads, 
            dropout = dropout_attn,
        )
        self.ln_2 = nn.LayerNorm(dim)
        self.FFNN = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_FFNN),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_FFNN)
        )

    def forward(self, x, seqlens):
        z = self.ln_1(x)
        x = x + self.self_attention(z, seqlens)
        y = self.ln_2(x)
        x = x + self.FFNN(y)
        
        return x
