from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.models.normalization import LayerNorm
from diffusers.utils import logging
from einops import repeat
import math

from ..embeddings import FrequencyPositionalEmbedding
from ..transformers.tripo2_transformer import DiTBlock
from ...utils import fps

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class Tripo2Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
        is_learned_queries: bool = False,
        sample_tokens: int = 32,
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
        fps: bool = False,
        is_miche: bool = False,
    ):
        super().__init__()

        self.fps = fps
        if fps and not is_learned_queries:
            self.embedder = FrequencyPositionalEmbedding(
                num_freqs=embed_frequency,
                logspace=True,
                input_dim=3,
                include_pi=embed_include_pi,
            )
            self.proj_k = nn.Linear(3+self.embedder.out_dim, dim, bias=True)
            self.proj_in = nn.Linear(in_channels-3+self.embedder.out_dim, dim, bias=True)
        else:
            self.proj_in = nn.Linear(in_channels, dim, bias=True)
        self.output_channels = dim
        self.is_miche = is_miche
        init_scale = 0.25 * math.sqrt(1.0 / dim)
        init_linear(self.proj_in, init_scale)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    use_cross_attention_2=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
        )
        self.norm_out = LayerNorm(dim)
        self.is_learned_queries = is_learned_queries
        if is_learned_queries:
            self.learned_queries = nn.Parameter(torch.randn(sample_tokens, dim) * 0.02)

    def forward(self, sample_1: torch.Tensor, sample_2: torch.Tensor, num_tokens: int=1024):
        if self.is_learned_queries or not self.fps:
            hidden_states = self.proj_in(sample_1) if not self.is_learned_queries else repeat(self.learned_queries[:sample_1.shape[1], :], 'n d -> b n d', b=sample_1.shape[0])
            encoder_hidden_states = self.proj_in(sample_2)
        else:
            x_q, x_kv = self.get_qkv(x=sample_1, num_tokens=num_tokens)
            hidden_states = self.proj_k(x_q)
            encoder_hidden_states = self.proj_in(x_kv)

        if not self.is_miche:
            for layer, block in enumerate(self.blocks):
                if layer == 0:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states=encoder_hidden_states
                    )
                else:
                    hidden_states = block(hidden_states)
        else:
            for layer, block in enumerate(self.blocks):
                if layer == 0:
                    hidden_states = block(hidden_states, encoder_hidden_states)
                else:
                    hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 1024, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 1024.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def get_qkv(self, x: torch.Tensor, num_tokens: int = 1024, seed: Optional[int] = None):
        positions, features = x[..., :3], x[..., 3:]
        x_kv = torch.cat([self.embedder(positions), features], dim=-1)

        sampled_x = self._sample_features(x, num_tokens, seed)
        positions, features = (
            sampled_x[..., :3],
            sampled_x[..., 3:],
        )
        x_q = torch.cat([self.embedder(positions), features], dim=-1)
        return x_q, x_kv


class Tripo2Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 16,
        grad_type: str = "analytical",
        grad_interval: float = 0.001,
        is_miche: bool = False,
    ):
        super().__init__()

        if grad_type not in ["numerical", "analytical"]:
            raise ValueError(f"grad_type must be one of ['numerical', 'analytical']")
        self.grad_type = grad_type
        self.grad_interval = grad_interval
        self.is_miche = is_miche
        
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    use_cross_attention_2=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
        )
        self.proj_query = nn.Linear(in_channels, dim, bias=True)

        self.norm_out = LayerNorm(dim)
        self.proj_out = nn.Linear(dim, out_channels, bias=True)
        self.sigmoid = nn.Sigmoid()
        init_scale = 0.25 * math.sqrt(1.0 / dim)
        init_linear(self.proj_query, init_scale)
        init_linear(self.proj_out, init_scale)

    def forward(
        self,
        sample: torch.Tensor,
        queries: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is None:
            hidden_states = sample
            for _, block in enumerate(self.blocks[:-1]):
                hidden_states = block(hidden_states)
            kv_cache = hidden_states
        # query grid logits by cross attention
        q = self.proj_query(queries)
        if self.is_miche:
            l = self.blocks[-1](q, kv_cache)
        else:
            l = self.blocks[-1](q, encoder_hidden_states=kv_cache)
        logits = self.proj_out(self.norm_out(l))

        logits = self.sigmoid(logits)
        assert kv_cache is not None
        return logits, kv_cache