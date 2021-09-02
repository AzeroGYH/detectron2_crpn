# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn, cat
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY

class ChannelAttention(nn.Module):
    def __init__(self, gate_channels=1024, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x): 
        channel_att_raw = self.mlp(x)
        scale = torch.sigmoid(channel_att_raw)
        return x * scale

def build_inner_attention_fc_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = "FastRCNNInnerAttentionFCHead"
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNInnerAttentionFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.ChannelAttention_0 = ChannelAttention(
            1024, reduction_ratio=16, pool_types=['avg', 'max'])
        self.ChannelAttention_1 = ChannelAttention(
            1024, reduction_ratio=16, pool_types=['avg', 'max'])
        self.ChannelAttentions = [self.ChannelAttention_0, self.ChannelAttention_1]
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x):
        if torch.numel(x) == 0:
            return x
        # for reg
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for fc, ChannelAttention in zip(self.fcs, self.ChannelAttentions):
                x = F.relu(fc(x))
                # ChannelAttention modelling
                x = ChannelAttention(x)
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

def build_inner_attention_fc_combine_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = "FastRCNNInnerAttentionFCCombineHead"
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNInnerAttentionFCCombineHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm=""
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.ChannelAttention_0 = ChannelAttention(
            1024, reduction_ratio=16, pool_types=['avg', 'max'])
        self.ChannelAttention_1 = ChannelAttention(
            1024, reduction_ratio=16, pool_types=['avg', 'max'])
        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x):
        if torch.numel(x) == 0:
            return x
        # for reg
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for fc in self.fcs:
                x = F.relu(fc(x))
        x_0 = self.ChannelAttention_0(x)
        x_1 = self.ChannelAttention_1(x)
        return x_0, x_1

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


class SelsaAggregator(nn.Module):
    """Selsa aggregator module.

    This module is proposed in "Sequence Level Semantics Aggregation for Video
    Object Detection". `SELSA <https://arxiv.org/abs/1907.06390>`_.

    Args:
        in_channels (int): The number of channels of the features of
            proposal.
        num_attention_blocks (int): The number of attention blocks used in
            selsa aggregator module. Defaults to 16.
    """

    def __init__(self, in_channels, num_attention_blocks=16):
        super(SelsaAggregator, self).__init__()
        self.fc_embed = nn.Linear(in_channels, in_channels)
        self.ref_fc_embed = nn.Linear(in_channels, in_channels)
        self.fc = nn.Linear(in_channels, in_channels)
        self.ref_fc = nn.Linear(in_channels, in_channels)
        self.num_attention_blocks = num_attention_blocks

    def forward(self, x, ref_x, mask=None):
        """Aggregate the features `ref_x` of reference proposals.

        The aggregation mainly contains two steps:
        1. Use multi-head attention to computing the weight between `x` and
        `ref_x`.
        2. Use the normlized (i.e. softmax) weight to weightedly sum `ref_x`.

        Args:
            x (Tensor): of shape [N, C]. N is the number of key frame
                proposals.
            ref_x (Tensor): of shape [M, C]. M is the number of reference frame
                proposals.

        Returns:
            Tensor: The aggregated features of key frame proposals with shape
            [N, C].
        """
        roi_n = x.shape[0]
        ref_roi_n = ref_x.shape[0]

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               -1).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       -1).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        
        mask = mask.repeat(self.num_attention_blocks, 1, 1)
        # print("mask", mask)
        # weights = weights.masked_fill(mask == 0, -1e9)
        # print("weights",weights.shape)
        weights = weights.softmax(dim=2)
        # print("weights", weights)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   -1).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, -1))
        return x_new

def build_selsa_attention_fc_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = "FastRCNNSelsaFCHead"
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNSelsaFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="",
        # num_fc=2, fc_dim=1024
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.aggregators = nn.ModuleList()
        for i in range(len(fc_dims)):
            self.aggregators.append(SelsaAggregator(in_channels=fc_dims[i]))
        self.inplace_false_relu = nn.ReLU(inplace=False)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x, mask):
        if torch.numel(x) == 0:
            return x
        # for reg
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for fc, aggregator in zip(self.fcs, self.aggregators):
                x = fc(x)
                x = x + aggregator(x, x, mask)
                x = self.inplace_false_relu(x)
        
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])
        
def build_selsa_inner_attention_fc_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = "FastRCNNSelsaInnerAttentionFCHead"
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNSelsaInnerAttentionFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, input_shape: ShapeSpec, *, conv_dims: List[int], fc_dims: List[int], conv_norm="",
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.aggregators = nn.ModuleList()
        self.ChannelAttentions = nn.ModuleList()
        for i in range(len(fc_dims)):
            self.aggregators.append(SelsaAggregator(in_channels=fc_dims[i]))
            self.ChannelAttentions.append(ChannelAttention(gate_channels=fc_dims[i], reduction_ratio=16))
        self.inplace_false_relu = nn.ReLU(inplace=False)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_BOX_HEAD.NORM,
        }

    def forward(self, x, mask):
        if torch.numel(x) == 0:
            return x
        # for reg
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for fc, aggregator, innerattenion in zip(self.fcs, self.aggregators, self.ChannelAttentions):
                x = fc(x)
                x = x + aggregator(x, x, mask)
                x = innerattenion(x)
                x = self.inplace_false_relu(x)
        
        return x

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])