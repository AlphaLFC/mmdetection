#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Wed Dec 18 2019 
@Time    : 23:15:53
@File    : dual_focal_loss.py.py
@Author  : alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def dual_focal_loss(pred, label, reduction='mean'):
    assert reduction in ['none', 'mean', 'sum']
    loss = torch.abs(label - pred) + F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


def _random_mask(tensor, percent):
    assert percent > 0
    return (torch.rand_like(tensor) < percent).float()


def balanced_dual_focal_loss(pred, label, neg_pos_ratio=4, least_neg_pct=0.05, reduction='mean'):
    assert reduction in ['none', 'mean', 'sum']
    mask_pos = (label > 0).float()
    rand_pct = mask_pos.sum() / mask_pos.nelement()
    mask = 1.0 + ((_random_mask(label, rand_pct * (neg_pos_ratio + 1)).clamp(least_neg_pct) + mask_pos) > 0).float()
    loss = dual_focal_loss(pred, label, reduction='none') * mask
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        return loss.sum()


@LOSSES.register_module
class DualFocalLoss(nn.Module):

    def __init__(self,
                 balance_sample=True,
                 neg_pos_ratio=4,
                 least_neg_pct=0.05,
                 use_one_hot_label=True,
                 num_classes=81,
                 reduction='mean',
                 loss_weight=1.0):
        super(DualFocalLoss, self).__init__()
        self.balance_sample = balance_sample
        self.neg_pos_ratio = neg_pos_ratio
        self.least_neg_pct = least_neg_pct
        self.use_one_hot_label = use_one_hot_label
        self.num_classes = num_classes
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        if self.use_one_hot_label:
            label = F.one_hot(label, num_classes=self.num_classes)[..., 1:]
        if self.balance_sample:
            return self.loss_weight * balanced_dual_focal_loss(
                pred, label,
                neg_pos_ratio=self.neg_pos_ratio,
                least_neg_pct=self.least_neg_pct,
                reduction=self.reduction
            )
        else:
            return self.loss_weight * dual_focal_loss(pred, label, reduction=self.reduction)