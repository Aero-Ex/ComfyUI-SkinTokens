# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional, Union
from einops import repeat
import math
import random
import time
import numpy as np

from ..modules import checkpoint
from ..modules.embedder import FourierEmbedder
from ..modules.distributions import DiagonalGaussianDistribution
from ..modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)
from ...utils.misc import use_flash3

from .tsal_base import ShapeAsLatentModule
from .loss import logits_to_sdf

from ....utils import fps

class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 query_method: bool = False,
                 use_full_input: bool = True,
                 token_num: int = 256,
                 no_query: bool=False):

        super().__init__()

        self.query_method = query_method
        self.token_num = token_num
        self.use_full_input = use_full_input

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        if no_query:
            self.query = None
        else:
            self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """
        if self.query_method:
            token_num = self.num_latents
            bs = pc.shape[0] #pc [10, 204800, 3]
            data = self.fourier_embedder(pc) #[10, 204800, 51]
            if feats is not None: #[10, 204800, 3]
                data = torch.cat([data, feats], dim=-1)
            data = self.input_proj(data) #[10, 204800, 768]

            query = repeat(self.query, "m c -> b m c", b=bs) #[10, 257, 768]

            latents = self.cross_attn(query, data)
            latents = self.self_attn(latents)

            if self.ln_post is not None:
                latents = self.ln_post(latents)

            pre_pc = None
        else:

            if isinstance(self.token_num, int):
                token_num = self.token_num
            else:
                token_num = random.choice(self.token_num)
            # print(token_num,'-----------------------', flush=True)

            if self.training:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed=0)
            ind = rng.choice(pc.shape[1], token_num * 4, replace=token_num * 4 > pc.shape[1])

            pre_pc = pc[:,ind,:]
            pre_feats = feats[:,ind,:]


            B, N, D = pre_pc.shape #[10, 204800, 3]            
            C = pre_feats.shape[-1]
            ###### fps
            pos = pre_pc.view(B*N, D)
            pos_feats = pre_feats.view(B*N, C)
            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)

            # ratio = 1.0 * token_num / N
            idx = fps(pos, batch, ratio=1. / 4, random_start=self.training)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)

            sampled_feats = pos_feats[idx]
            sampled_feats = sampled_feats.view(B, -1, C)

            ######
            if self.use_full_input:
                data = self.fourier_embedder(pc) #[B, 20480, 51]
            else:
                data = self.fourier_embedder(pre_pc) # [B, 4 * token_num, 51]

            if feats is not None: #[10, 204800, 3]
                if not self.use_full_input:
                    feats = pre_feats
                data = torch.cat([data, feats], dim=-1) #[10, 204800, 54]
            data = self.input_proj(data) #[10, 204800, 768]

            # print(data.shape)

            sampled_data = self.fourier_embedder(sampled_pc) #[10, 256, 51]
            if feats is not None: #[10, 256, 3]
                sampled_data = torch.cat([sampled_data, sampled_feats], dim=-1) #[10, 256, 54]
            sampled_data = self.input_proj(sampled_data) #[10, 256, 768]

            latents = self.cross_attn(sampled_data, data) #[10, 256, 768]
            latents = self.self_attn(latents)

            if self.ln_post is not None:
                latents = self.ln_post(latents)

            pre_pc = torch.cat([pre_pc, pre_feats], dim=-1)

        return latents, pc, token_num, pre_pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)


class CrossAttentionDecoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False,
                 mlp_width_scale: int = 4,
                 supervision_type: str = 'occupancy'):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder
        self.supervision_type = supervision_type

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            mlp_width_scale=mlp_width_scale,
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)
        if self.supervision_type == 'occupancy-sdf':
            self.output_proj_sdf = nn.Linear(width, out_channels, device=device, dtype=dtype)



    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        if next(self.query_proj.parameters()).dtype == torch.float16:
            queries = queries.half()
            latents = latents.half()
        # print(f"queries: {queries.dtype}, {queries.device}")
        # print(f"latents: {latents.dtype}, {latents.device}"z)
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x_1 = self.output_proj(x)
        if self.supervision_type == 'occupancy-sdf':
            x_2 = self.output_proj_sdf(x)
            return x_1, x_2
        else:
            return x_1

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 decoder_width: Optional[int] = None,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 query_method: bool = False,
                 token_num: int = 256,
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 use_full_input: bool = True,
                 freeze_encoder: bool = False,
                 decoder_mlp_width_scale: int = 4,
                 residual_kl: bool = False,
                 ):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        assert grad_type in ["numerical", "analytical"]
        self.grad_type = grad_type
        self.grad_interval = grad_interval
        self.supervision_type = supervision_type
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
            query_method=query_method,
            use_full_input=use_full_input,
            token_num=token_num
        )

        self.embed_dim = embed_dim
        self.residual_kl = residual_kl
        if decoder_width is None:
            decoder_width = width
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, decoder_width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
            if self.residual_kl:
                assert self.post_kl.out_features % self.post_kl.in_features == 0
                assert self.pre_kl.in_features % self.pre_kl.out_features == 0 
        else:
            self.latent_shape = (num_latents, width)

        print("decoder width = ", decoder_width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=decoder_width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=decoder_width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint,
            supervision_type=supervision_type,
            mlp_width_scale=decoder_mlp_width_scale
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.pre_kl.parameters():
                p.requires_grad = False
            print("freeze encoder and pre kl")

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """

        latents, center_pos = self.encoder(pc, feats)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            if self.residual_kl:
                B, N = latents.shape[:2]
                moments = moments + latents.view(B, N, -1, self.pre_kl.out_features).mean(dim=-2)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        if self.residual_kl:
            latents = latents.repeat_interleave(self.post_kl.out_features // self.post_kl.in_features, dim=-1) + self.post_kl(latents)
        else:
            latents = self.post_kl(latents)
        
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor, grad: bool = False):
        # logits = self.geo_decoder(queries, latents).squeeze(-1)
        if grad:
          # with torch.autocast(device_type="cuda", dtype=torch.float32):
            if self.grad_type == "numerical":
                raise NotImplementedError
                interval = self.grad_interval
                # print('grad interval = ', interval)
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=queries.device)[None, :]
                    res_p = self.geo_decoder(queries + offset_tensor, latents)[..., 0]
                    res_n = self.geo_decoder(queries - offset_tensor, latents)[..., 0]
                    grad_value.append((res_p - res_n) / (2 * interval))
                grad_value = torch.stack(grad_value, dim=-1)
            else:
                # print("auto grad")
                queries_d = torch.clone(queries)
                queries_d.requires_grad = True
                with torch.enable_grad():
                    with use_flash3.disable_flash3():
                        logits = self.geo_decoder(queries_d, latents)
                        if self.supervision_type == "sigmoid-sdf":
                            sdfs = logits_to_sdf(logits)
                        grad_value = torch.autograd.grad(sdfs, [queries_d],
                                                        grad_outputs=torch.ones_like(sdfs),
                                                        create_graph=self.geo_decoder.training)[0]
        else:
            logits = self.geo_decoder(queries, latents)
            grad_value = None
                
        return logits, grad_value

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """

        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior


class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[str],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 decoder_width: Optional[int] = None,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 query_method: bool = False,
                 use_full_input: bool = True,
                 token_num: int = 256,
                 freeze_encoder: bool = False,
                 decoder_mlp_width_scale: int = 4,
                 residual_kl: bool = False,
                ):

        MAP_DTYPE = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        if dtype is not None:
            dtype = MAP_DTYPE[dtype]
        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            decoder_width=decoder_width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
            supervision_type=supervision_type,
            grad_type=grad_type,
            grad_interval=grad_interval,
            query_method=query_method,
            token_num=token_num,
            use_full_input=use_full_input,
            freeze_encoder=freeze_encoder,
            decoder_mlp_width_scale=decoder_mlp_width_scale,
            residual_kl=residual_kl,
        )

        self.width = width

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True,
               only_shape: bool=False):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, c]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor)
            kl_embed (torch.FloatTensor):
            posterior (DiagonalGaussianDistribution or None):
        """

        shape_embed, latents, token_num, pre_pc = self.encode_latents(pc, feats)
        if only_shape:
            return shape_embed
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)

        return shape_embed, kl_embed, posterior, token_num, pre_pc

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):

        x, _, token_num, pre_pc = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        # latents = x[:, 1:]
        # use all tokens
        latents = x

        return shape_embed, latents, token_num, pre_pc

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            if self.residual_kl:
                B, N = latents.shape[:2]
                moments = moments + latents.view(B, N, -1, self.pre_kl.out_features).mean(dim=-2)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor): [B, projection_dim]
            logits (torch.FloatTensor): [B, M]
            posterior (DiagonalGaussianDistribution or None).

        """

        shape_embed, kl_embed, posterior, token_num, pre_pc  = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(kl_embed)
        logits, grad = self.query_geometry(volume_queries, latents)

        return shape_embed, logits, posterior, token_num, pre_pc, grad

#####################################################
# a simplified verstion of perceiver encoder
#####################################################

class ShapeAsLatentPerceiverEncoder(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[Union[torch.dtype, str]],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 supervision_type: str = 'occupancy',
                 query_method: bool = False,
                 token_num: int = 256,
                 grad_type: str = "numerical",
                 grad_interval: float = 0.005,
                 use_full_input: bool = True,
                 freeze_encoder: bool = False,
                 residual_kl: bool = False,
                 ):

        super().__init__()


        MAP_DTYPE = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }

        if dtype is not None and isinstance(dtype, str):
            dtype = MAP_DTYPE[dtype]

        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        assert grad_type in ["numerical", "analytical"]
        self.grad_type = grad_type
        self.grad_interval = grad_interval
        self.supervision_type = supervision_type
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint,
            query_method=query_method,
            use_full_input=use_full_input,
            token_num=token_num,
            no_query=True,
        )

        self.embed_dim = embed_dim
        self.residual_kl = residual_kl
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("freeze encoder")
        self.width = width

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):

        x, _, token_num, pre_pc = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        latents = x

        return shape_embed, latents, token_num, pre_pc

    def forward(self):
        raise NotImplementedError()