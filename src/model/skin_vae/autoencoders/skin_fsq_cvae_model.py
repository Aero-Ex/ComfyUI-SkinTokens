from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.modeling_utils import ModelMixin
from einops import repeat
import math

from ..attention_processor import Tripo2AttnProcessor2_0
from ..embeddings import FrequencyPositionalEmbedding
from .autoencoder_kl_tripo2 import Tripo2Encoder, Tripo2Decoder
from .FSQ import FSQ
from .SimVQ import SimVQ1D

from ...utils import fps

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class SkinFSQCVAEModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 3,
        latent_channels: int = 64,
        num_attention_heads: int = 8,
        width_encoder: int = 512,
        width_decoder: int = 1024,
        num_layers_encoder: int = 8,
        num_layers_decoder: int = 16,
        embedding_type: str = "frequency",
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
        sample_tokens: int = 32,
        **kwargs
    ):
        super().__init__()

        self.out_channels = 1

        if embedding_type == "frequency":
            self.embedder = FrequencyPositionalEmbedding(
                num_freqs=embed_frequency,
                logspace=True,
                input_dim=3,
                include_pi=embed_include_pi,
                use_pmpe=kwargs.get('use_pmpe', False),
            )
        else:
            raise NotImplementedError(
                f"Embedding type {embedding_type} is not supported."
            )

        self.is_learned_queries = kwargs['is_learned_queries']

        is_miche = kwargs.get('is_miche', False)
        self.encoder = Tripo2Encoder(
            in_channels=in_channels + self.embedder.out_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
            is_learned_queries=self.is_learned_queries,
            sample_tokens=sample_tokens,
            is_miche=is_miche,
        )

        self.cond_encoder = Tripo2Encoder(
            in_channels=cond_channels + self.embedder.out_dim,
            dim=width_encoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_encoder,
            is_miche=is_miche,
        )

        self.decoder = Tripo2Decoder(
            in_channels=self.embedder.out_dim + self.cond_channels,
            out_channels=self.out_channels,
            dim=width_decoder,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers_decoder,
            is_miche=is_miche,
        )

        self.cond_quant = nn.Linear(width_encoder, latent_channels, bias=True)

        self.quant = nn.Linear(width_encoder, latent_channels, bias=True)
        self.post_quant = nn.Linear(latent_channels, width_decoder, bias=True)

        init_scale = 0.25 * math.sqrt(1.0 / width_encoder)
        init_linear(self.cond_quant, init_scale)
        init_linear(self.quant, init_scale)
        init_scale = 0.25 * math.sqrt(1.0 / latent_channels)
        init_linear(self.post_quant, init_scale)
        self.use_slicing = False
        self.slicing_length = 1
        if kwargs.get('FSQ_dict', None) is not None:
            self.FSQ = FSQ(**kwargs['FSQ_dict'])
        else:
            self.FSQ = SimVQ1D(**kwargs['SimVQ_dict'])

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(Tripo2AttnProcessor2_0())

    def enable_slicing(self, slicing_length: int = 1) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True
        self.slicing_length = slicing_length

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 128, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
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

    def get_qkv(self, x: torch.Tensor, num_tokens: int = 128, seed: Optional[int] = None, not_get_q: bool=False):
        positions, features = x[..., :3], x[..., 3:]
        x_kv = torch.cat([self.embedder(positions), features], dim=-1)

        if not_get_q:
            x_q = torch.zeros((x.shape[0], num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
        else:
            sampled_x = self._sample_features(x, num_tokens, seed)
            positions, features = (
                sampled_x[..., :3],
                sampled_x[..., 3:],
            )
            x_q = torch.cat([self.embedder(positions), features], dim=-1)
        return x_q, x_kv

    def _encode(
        self, x: torch.Tensor|None, cond: torch.Tensor|None, num_tokens: int = 128, cond_tokens: int = 128, seed: Optional[int] = None,
        return_z: bool=True, return_cond: bool=True,
    ):
        position_channels = 3
        if return_z:
            assert x is not None
            x_q, x_kv = self.get_qkv(x, num_tokens, seed, not_get_q=self.is_learned_queries)
            x = self.encoder(x_q, x_kv)
            x = self.quant(x)
        else:
            x = None

        if return_cond:
            assert cond is not None
            cond_q, cond_kv = self.get_qkv(cond, cond_tokens, seed)
            cond_embed = self.cond_encoder(cond_q, cond_kv)
            cond = self.cond_quant(cond_embed)
        else:
            cond = None

        return x, cond

    def _decode(
        self, z: torch.Tensor,
        cond: torch.Tensor,
        sampled_points: torch.Tensor,
        num_chunks: Optional[int] = None,
    ) -> torch.Tensor:
        xyz_samples = sampled_points
        z = self.post_quant(torch.cat([z, cond], dim=1))

        num_points = xyz_samples.shape[1]
        if num_chunks is None:
            num_chunks = num_points

        queries = sampled_points.to(z.device, dtype=z.dtype)
        positions, features = (
            queries[..., :3],
            queries[..., 3:],
        )

        kv_cache = None
        dec = []
        for i in range(0, num_points, num_chunks):
            queries = torch.cat([self.embedder(positions[:, i:i + num_chunks, :]), features[:, i:i + num_chunks, :]], dim=-1)
            z, kv_cache = self.decoder(z, queries, kv_cache)
            dec.append(z)

        return torch.cat(dec, dim=1)
    
    def compile_model(self):
        self.encoder = torch.compile(self.encoder)
        self.cond_encoder = torch.compile(self.cond_encoder)
        self.decoder = torch.compile(self.decoder)
    
    def forward(self, x: torch.Tensor):
        pass
