#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Mon Jun 24 2019 
@Time    : 21:57:52
@File    : coord_conv.py
@Author  : alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, offset=0.5):
        super(CoordConv2d, self).__init__(
            in_channels+2, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        assert 0 <= offset < 1
        self.offset = offset
        self.register_forward_pre_hook(self.add_location_maps)

    @staticmethod
    def add_location_maps(self, inputs):
        input, *_ = inputs
        N, C, H, W = input.shape
        dtype, device = input.dtype, input.device
        n = torch.zeros(N, dtype=dtype, device=device)
        c = torch.zeros(1, dtype=dtype, device=device)
        h = torch.arange(self.offset, H, dtype=dtype, device=device) / H
        w = torch.arange(self.offset, W, dtype=dtype, device=device) / W
        _, _, grid_h, grid_w = torch.meshgrid(n, c, h, w)
        self.coord_input = torch.cat([input, grid_h, grid_w], dim=1)

    def forward(self, input):
        return F.conv2d(self.coord_input,
                        self.weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)
