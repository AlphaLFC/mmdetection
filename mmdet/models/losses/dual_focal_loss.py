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


def dual_focal_loss_old(pred, label, reduction='mean', avg_factor=None):
    """
    :param pred:
    :param label:
    :param reduction:
    :param avg_factor:
    :return:
    """
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    loss = torch.abs(label - pred.sigmoid()) + F.binary_cross_entropy_with_logits(pred, label, reduction='none')
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        if avg_factor is None:
            return loss.mean()
        else:
            return loss.sum() / avg_factor
    else:
        return loss.sum()


def dual_focal_loss_proto(pred, label, reduction='mean', avg_factor=None):
    """
    :param pred: logits
    :param label: 0~1 floats
    :param reduction: 'none', 'sum', 'mean'
    :param avg_factor:
    :return: loss
    dfl = abs(label - pred.sigmoid()) - log(1 - abs(label - pred.sigmoid()))
    """
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    pred_sigmoid = pred.sigmoid()
    l1 = torch.abs(label - pred_sigmoid)
    loss = l1 - (1.0 - l1).log()
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        if avg_factor is None:
            return loss.mean()
        else:
            return loss.sum() / avg_factor
    else:
        return loss.sum()


def dual_focal_loss(pred, label, reduction='mean', avg_factor=None):
    """
    :param pred: logits
    :param label: 0~1 floats
    :param reduction: 'none', 'sum', 'mean'
    :param avg_factor:
    :return: loss
    dfl = abs(label - pred.sigmoid()) - log(1 - abs(label - pred.sigmoid()))
    """
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    pred_sigmoid = pred.sigmoid()
    l1 = torch.abs(label - pred_sigmoid)
    sigmoid_inv = 1.0 + (-pred).exp()
    item = torch.where(label > pred_sigmoid,
                       sigmoid_inv * (1 - label) + 1,
                       sigmoid_inv * (1 + label) - 1)
    loss = l1 - item.log() + sigmoid_inv.log()
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        if avg_factor is None:
            return loss.mean()
        else:
            return loss.sum() / avg_factor
    else:
        return loss.sum()


def _random_mask(tensor, percent):
    assert percent > 0
    return (torch.rand_like(tensor) < percent).float()


def balanced_dual_focal_loss(pred, label, neg_pos_ratio=4, least_neg_pct=0.05, reduction='mean'):
    assert reduction in ['none', 'mean', 'sum']
    label = label.type_as(pred)
    mask_pos = (label > 0).float()
    rand_pct = mask_pos.sum() / mask_pos.nelement()
    neg_pct = (rand_pct * (neg_pos_ratio + 1)).clamp(least_neg_pct)
    mask = 1.0 + ((_random_mask(label, neg_pct) + mask_pos) > 0).float()
    mask /= mask.mean()
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
        if self.use_one_hot_label:
            self.loss_weight = loss_weight * (num_classes - 1)
        else:
            self.loss_weight = loss_weight

    def forward(self, pred, label,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.use_one_hot_label:
            pred = pred.flatten()
            label = F.one_hot(label, num_classes=self.num_classes)[..., 1:].flatten()
        if self.balance_sample:
            return self.loss_weight * balanced_dual_focal_loss(
                pred, label,
                neg_pos_ratio=self.neg_pos_ratio,
                least_neg_pct=self.least_neg_pct,
                reduction=reduction
            )
        else:
            return self.loss_weight * dual_focal_loss(
                pred, label,
                reduction=reduction,
                avg_factor=avg_factor
            )