from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    SiLU,
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    convert_module_to_f16
)

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from dataclasses import dataclass

@dataclass
class UNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: th.FloatTensor


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class CondTimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, cond, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock, CondTimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, cond, emb):
        for layer in self:
            if isinstance(layer, CondTimestepBlock):
                x = layer(x, cond, emb)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class SPADEGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, eps = 1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16

        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        return x * (1 + gamma) + beta

class LLIGN(nn.Module):
    def __init__(self, norm_nc, label_nc, eps=1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False)  # 32/16
        self.eps = eps
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        
        # Initialize small class weight parameter
        self.small_class_weight = nn.Parameter(torch.ones(num_classes))
        
        # Define the small class indices
        self.small_class_indices = torch.tensor([3,6,7,10,14], dtype=torch.long)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap_resized = F.interpolate(segmap, size=x.size()[2:], mode='bicubic', align_corners=True)
        
        # Apply weights to small classes
        weights = self.small_class_weight[None, :, None, None]
        segmap_resized = segmap_resized * weights
        
        actv = self.mlp_shared(segmap_resized)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        return x * (1 + gamma) + beta

class AdaIN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.instance_norm = th.nn.InstanceNorm2d(num_features, affine=False, track_running_stats=False)

    def forward(self, x, alpha, gamma):
        assert x.shape[:2] == alpha.shape[:2] == gamma.shape[:2]
        norm = self.instance_norm(x)
        return alpha * norm + gamma

class RESAILGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, guidance_nc, eps = 1e-5):
        super().__init__()

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16

        # SPADE
        self.eps = eps
        nhidden = 128
        self.mask_mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mask_mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mask_mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)


        # Guidance

        self.conv_s = th.nn.Conv2d(label_nc, nhidden * 2, 3, 2)
        self.pool_s = th.nn.AdaptiveAvgPool2d(1)
        self.conv_s2 = th.nn.Conv2d(nhidden * 2, nhidden * 2, 1, 1)

        self.conv1 = th.nn.Conv2d(guidance_nc, nhidden, 3, 1, padding=1)
        self.adaIn1 = AdaIN(norm_nc * 2)
        self.relu1 = nn.ReLU()

        self.conv2 = th.nn.Conv2d(nhidden, nhidden, 3, 1, padding=1)
        self.adaIn2 = AdaIN(norm_nc * 2)
        self.relu2 = nn.ReLU()
        self.conv3 = th.nn.Conv2d(nhidden, nhidden, 3, 1, padding=1)

        self.guidance_mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.guidance_mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

        self.blending_gamma = nn.Parameter(th.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(th.zeros(1), requires_grad=True)
        self.norm_nc = norm_nc

    def forward(self, x, segmap, guidance):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        mask_actv = self.mask_mlp_shared(segmap)
        mask_gamma = self.mask_mlp_gamma(mask_actv)
        mask_beta = self.mask_mlp_beta(mask_actv)


        # Part 3. produce scaling and bias conditioned on feature guidance
        guidance = F.interpolate(guidance, size=x.size()[2:], mode='bilinear')

        f_s_1 = self.conv_s(segmap)
        c1 = self.pool_s(f_s_1)
        c2 = self.conv_s2(c1)

        f1 = self.conv1(guidance)

        f1 = self.adaIn1(f1, c1[:, : 128, ...], c1[:, 128:, ...])
        f2 = self.relu1(f1)

        f2 = self.conv2(f2)
        f2 = self.adaIn2(f2, c2[:, : 128, ...], c2[:, 128:, ...])
        f2 = self.relu2(f2)
        guidance_actv = self.conv3(f2)

        guidance_gamma = self.guidance_mlp_gamma(guidance_actv)
        guidance_beta = self.guidance_mlp_beta(guidance_actv)

        gamma_alpha = F.sigmoid(self.blending_gamma)
        beta_alpha = F.sigmoid(self.blending_beta)

        gamma_final = gamma_alpha * guidance_gamma + (1 - gamma_alpha) * mask_gamma
        beta_final = beta_alpha * guidance_beta + (1 - beta_alpha) * mask_beta
        out = x * (1 + gamma_final) + beta_final

        # apply scale and bias
        return out

class SPMGroupNorm(nn.Module):
    def __init__(self, norm_nc, label_nc, feature_nc, eps = 1e-5):
        super().__init__()
        print("use SPM")

        self.norm = nn.GroupNorm(32, norm_nc, affine=False) # 32/16

        # SPADE
        self.eps = eps
        nhidden = 128
        self.mask_mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.mask_mlp_gamma1 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mask_mlp_beta1 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

        self.mask_mlp_gamma2 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mask_mlp_beta2 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)


        # Feature
        self.feature_mlp_shared = nn.Sequential(
            nn.Conv2d(feature_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.feature_mlp_gamma1 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.feature_mlp_beta1 = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)


    def forward(self, x, segmap, guidance):
        # Part 1. generate parameter-free normalized activations
        x = self.norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        mask_actv = self.mask_mlp_shared(segmap)
        mask_gamma1 = self.mask_mlp_gamma1(mask_actv)
        mask_beta1 = self.mask_mlp_beta1(mask_actv)

        mask_gamma2 = self.mask_mlp_gamma2(mask_actv)
        mask_beta2 = self.mask_mlp_beta2(mask_actv)


        # Part 3. produce scaling and bias conditioned on feature guidance
        guidance = F.interpolate(guidance, size=x.size()[2:], mode='bilinear')
        feature_actv = self.feature_mlp_shared(guidance)
        feature_gamma1 = self.feature_mlp_gamma1(feature_actv)
        feature_beta1 = self.feature_mlp_beta1(feature_actv)

        gamma_final = feature_gamma1 * (1 + mask_gamma1) + mask_beta1
        beta_final = feature_beta1 * (1 + mask_gamma2) + mask_beta2

        out = x * (1 + gamma_final) + beta_final

        # apply scale and bias
        return out


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """

        return th.utils.checkpoint.checkpoint(self._forward, x ,emb)
        # return checkpoint(
        #     self._forward, (x, emb), self.parameters(), self.use_checkpoint
        # )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)#.type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class SDMResBlock(CondTimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        c_channels=3,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        SPADE_type = "LLIGN",
        guidance_nc = None
    ):
        super().__init__()
        self.channels = channels
        self.guidance_nc = guidance_nc
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.SPADE_type = SPADE_type
        if self.SPADE_type == "LLIGN":
            self.in_norm = LLIGN(channels, c_channels)
        elif self.SPADE_type == "SPADE":
            self.in_norm = SPADEGroupNorm(channels, c_channels)
        elif self.SPADE_type == "RESAIL":
            self.in_norm = RESAILGroupNorm(channels, c_channels, guidance_nc)
        elif self.SPADE_type == "SPM":
            self.in_norm = SPMGroupNorm(channels, c_channels, guidance_nc)
        self.in_layers = nn.Sequential(
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        if self.SPADE_type == "LLIGN":
            self.out_norm = LLIGN(self.out_channels, c_channels)
        elif self.SPADE_type == "SPADE":
            self.out_norm = SPADEGroupNorm(self.out_channels, c_channels)
        elif self.SPADE_type == "RESAIL":
            self.out_norm = RESAILGroupNorm(self.out_channels, c_channels, guidance_nc)
        elif self.SPADE_type == "SPM":
            self.out_norm = SPMGroupNorm(self.out_channels, c_channels, guidance_nc)

        self.out_layers = nn.Sequential(
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, cond, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return th.utils.checkpoint.checkpoint(self._forward, x, cond, emb)
        # return checkpoint(
        #     self._forward, (x, cond, emb), self.parameters(), self.use_checkpoint
        # )

    def _forward(self, x, cond, emb):
        if self.SPADE_type == "RESAIL" or self.SPADE_type == "SPM":
            assert self.guidance_nc is not None, "Please set guidance_nc when you use RESAIL"
            guidance = x[: ,x.shape[1] - self.guidance_nc:, ...]
        else:
            guidance = None
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            if self.SPADE_type == "LLIGN" or self.SPADE_type == "SPADE":
                h = self.in_norm(x, cond)
            elif self.SPADE_type == "RESAIL" or self.SPADE_type == "SPM":
                h = self.in_norm(x, cond, guidance)

            h = in_rest(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            if self.SPADE_type == "LLIGN" or self.SPADE_type == "SPADE":
                h = self.in_norm(x, cond)
            elif self.SPADE_type == "RESAIL" or self.SPADE_type == "SPM":
                h = self.in_norm(x, cond, guidance)
            h = self.in_layers(h)

        emb_out = self.emb_layers(emb)#.type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            if self.SPADE_type == "LLIGN" or self.SPADE_type == "SPADE":
                h = self.out_norm(h, cond)
            elif self.SPADE_type == "RESAIL" or self.SPADE_type == "SPM":
                h = self.out_norm(h, cond, guidance)

            h = h * (1 + scale) + shift
            h = self.out_layers(h)
        else:
            h = h + emb_out
            if self.SPADE_type == "LLIGN" or self.SPADE_type == "SPADE":
                h = self.out_norm(h, cond)
            elif self.SPADE_type == "RESAIL" or self.SPADE_type == "SPM":
                h = self.out_norm(x, cond, guidance)

            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return th.utils.checkpoint.checkpoint(self._forward, x)
        #return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(ModelMixin, ConfigMixin):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    _supports_gradient_checkpointing = True
    @register_to_config
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=True,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        mask_emb="resize",
        SPADE_type="LLIGN",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.sample_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.mask_emb = mask_emb

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))] #ch=256
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                #print(ds)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        self.middle_block = TimestepEmbedSequential(
            SDMResBlock(
                ch,
                time_embed_dim,
                dropout,
                c_channels=num_classes if mask_emb == "resize" else num_classes*4,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            SDMResBlock(
                ch,
                time_embed_dim,
                dropout,
                c_channels=num_classes if mask_emb == "resize" else num_classes*4 ,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                #print(ch, ich)
                layers = [
                    SDMResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        c_channels=num_classes if mask_emb == "resize" else num_classes*4,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        SPADE_type=SPADE_type,
                        guidance_nc = ich,
                    )
                ]
                ch = int(model_channels * mult)
                #print(ds)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        SDMResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            c_channels=num_classes if mask_emb == "resize" else num_classes*4,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )
    def _set_gradient_checkpointing(self, module, value=False):
        #if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
        module.gradient_checkpointing = value
    def forward(self, x, y=None, timesteps=None ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        if not th.is_tensor(timesteps):
            timesteps = th.tensor([timesteps], dtype=th.long, device=x.device)
        elif th.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)

        timesteps = timestep_embedding(timesteps, self.model_channels).type(x.dtype).to(x.device)
        emb = self.time_embed(timesteps)

        y = y.type(self.dtype)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            # input_blocks have no any opts for y
            h = module(h, y, emb)
            #print(h.shape)
            hs.append(h)

        h = self.middle_block(h, y, emb)

        for module in self.output_blocks:
            temp = hs.pop()

            #print("before:", h.shape, temp.shape)
            # copy padding to match the downsample size
            if h.shape[2] != temp.shape[2]:
                p1d = (0, 0, 0, 1)
                h = F.pad(h, p1d, "replicate")

            if h.shape[3] != temp.shape[3]:
                p2d = (0, 1, 0, 0)
                h = F.pad(h, p2d, "replicate")
            #print("after:", h.shape, temp.shape)

            h = th.cat([h, temp], dim=1)
            h = module(h, y, emb)

        h = h.type(x.dtype)
        return UNet2DOutput(sample=self.out(h))


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, cond, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, cond, timesteps, **kwargs)


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)

