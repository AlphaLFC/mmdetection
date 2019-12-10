#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by PyCharm.

@Date    : Tue Dec 10 2019 
@Time    : 21:15:58
@File    : balanced_bce_loss.py
@Author  : alpha
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOSSES


def _random_choice(tensor, num):
    flattened_tensor = tensor.flatten()
    rand_inds = torch.randperm(flattened_tensor.nelement())[:num]
    return flattened_tensor[rand_inds]


def balanced_bce_with_logits_loss(pred, label,
                                  rand_neg_ratio=3,
                                  least_neg_percent=0.05,
                                  use_ohem=False,
                                  ohem_neg_ratio=1,
                                  ohem_pos_ratio=0.5,
                                  reduction='mean'):
    assert reduction in ['mean', 'sum'], "'none' mode not supported in this loss."
    flattened_pred = pred.flatten()
    flattened_label = label.flatten()
    pos_inds = flattened_label.nonzero().reshape(-1).cpu()
    neg_inds = (flattened_label == 0).nonzero().reshape(-1).cpu()
    rand_neg_num = max(
        int(len(pos_inds) * rand_neg_ratio),
        int(flattened_label.nelements() * least_neg_percent)
    )
    if use_ohem:
        ohem_neg_num = int(len(pos_inds) * ohem_neg_ratio)
        ohem_pos_num = int(len(pos_inds) * ohem_pos_ratio)
        neg_pred, neg_label = flattened_pred[neg_inds], flattened_label[neg_inds]
        pos_pred, pos_label = flattened_pred[pos_inds], flattened_label[pos_inds]
        sorted_neg_inds = neg_pred.argsort(descending=True).cpu()
        sorted_pos_inds = pos_pred.argsort().cpu()
        ohem_neg_inds = sorted_neg_inds[:ohem_neg_num]
        ohem_pos_inds = sorted_pos_inds[:ohem_pos_num]
        rand_sorted_neg_inds = _random_choice(sorted_neg_inds, rand_neg_num)
        balanced_neg_inds = torch.cat((ohem_neg_inds, rand_sorted_neg_inds)).sort()[0]
        balanced_pos_inds = torch.cat((sorted_pos_inds, ohem_pos_inds)).sort()[0]
        balanced_pred = torch.cat((pos_pred[balanced_pos_inds], neg_pred[balanced_neg_inds]))
        balanced_label = torch.cat((pos_label[balanced_pos_inds], neg_label[balanced_neg_inds]))
        loss = F.binary_cross_entropy_with_logits(balanced_pred, balanced_label, reduction=reduction)
    else:
        rand_neg_inds = _random_choice(neg_inds, rand_neg_num)
        balanced_inds = torch.cat((pos_inds, rand_neg_inds)).sort()[0]
        loss = F.binary_cross_entropy_with_logits(
            flattened_pred[balanced_inds],
            flattened_label[balanced_inds],
            reduction=reduction
        )
    return loss


@LOSSES.register_module
class BalancedBCEWithLogitsLoss(nn.Module):

    def __init__(self,
                 rand_neg_ratio=3,
                 least_neg_percent=0.05,
                 use_ohem=False,
                 ohem_neg_ratio=1,
                 ohem_pos_ratio=0.5,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedBCEWithLogitsLoss, self).__init__()
        self.rand_neg_ratio = rand_neg_ratio
        self.least_neg_percent = least_neg_percent
        self.use_ohem = use_ohem
        self.ohem_neg_ratio = ohem_neg_ratio
        self.ohem_pos_ratio = ohem_pos_ratio
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, label):
        return self.loss_weight * balanced_bce_with_logits_loss(
            pred, label,
            rand_neg_ratio=self.rand_neg_ratio,
            least_neg_percent=self.least_neg_percent,
            use_ohem=self.use_ohem,
            ohem_neg_ratio=self.ohem_neg_ratio,
            ohem_pos_ratio=self.ohem_pos_ratio,
            reduction=self.reduction
        )
