# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Wei Yang (platero.yang@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSELoss, self).__init__()
        self.MSEcriterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.MSEcriterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.MSEcriterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsMSEandBCELoss(nn.Module):
    def __init__(self, use_target_weight=False):
        super(JointsMSEandBCELoss, self).__init__()
        self.MSEcriterion = nn.MSELoss(reduction='mean')
        self.BCEcriterion = nn.BCEWithLogitsLoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_pred_tuple = heatmaps_pred.split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        heatmaps_gt_tuple = heatmaps_gt.split(1, 1)
        
        sum_pred = torch.sum(heatmaps_pred, axis=2)
        sum_gt = torch.sum(heatmaps_gt, axis=2)
        loss = 0 #0.00005 * self.BCEcriterion(sum_pred, sum_gt) # init loss with false detections (non empty masks) sum all the pixels in the image

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred_tuple[idx].squeeze()
            heatmap_gt = heatmaps_gt_tuple[idx].squeeze()
            if self.use_target_weight:
                loss += 1 * self.MSEcriterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
                loss += 0.0005 * self.BCEcriterion(
                    heatmap_pred.mul(torch.where(target_weight[:, idx]==0,1.,0)),
                    heatmap_gt.mul(torch.zeros_like(target_weight[:, idx]))
                ) 
            else:
                loss += 0.5 * self.MSEcriterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsMSEandBCEandDICELoss(nn.Module):
    def __init__(self):
        super(JointsMSEandBCEandDICELoss, self).__init__()
        self.MSEcriterion = nn.MSELoss(reduction='mean')
        self.BCEcriterion = nn.BCEWithLogitsLoss()
        self.DICEcriterion = diceloss()

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_pred_tuple = heatmaps_pred.split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        heatmaps_gt_tuple = heatmaps_gt.split(1, 1)
        
        sum_pred = torch.sum(heatmaps_pred, axis=2)
        sum_gt = torch.sum(heatmaps_gt, axis=2)
        loss_mse = 0
        loss_bce = 0
        loss_dice = self.DICEcriterion(
            heatmaps_pred,
            heatmaps_gt
        )
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred_tuple[idx].squeeze()
            heatmap_gt = heatmaps_gt_tuple[idx].squeeze()
            loss_mse += self.MSEcriterion(
                heatmap_pred,
                heatmap_gt
            )
            loss_bce += self.BCEcriterion(
                heatmap_pred.mul(torch.where(target_weight[:, idx]==0,1.,0)),
                heatmap_gt.mul(torch.zeros_like(target_weight[:, idx]))
            )
        loss = 1*loss_mse + 0*loss_bce + 0*loss_dice
        return loss / num_joints


class JointsBCELoss(nn.Module):
    def __init__(self):
        super(JointsBCELoss, self).__init__()
        self.BCEcriterion = nn.BCEWithLogitsLoss()

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        
        loss = 0 #self.BCEcriterion(vis_out, target_weight) # init loss with false detections (non empty masks)

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
        
            loss += 0.005 * self.BCEcriterion(
                heatmap_pred,
                heatmap_gt
            ) 

        return loss / num_joints

class diceloss(torch.nn.Module):
    '''
    Based on:
    - https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
    - https://github.com/Project-MONAI/MONAI/blob/dev/monai/losses/dice.py 
    '''
    def init(self):
        super(diceLoss, self).init()
    def forward(self, pred, target):
        smooth = 1e-5

        sum_pred = torch.sum(pred, axis=2)
        sum_gt = torch.sum(target, axis=2)

        intersection = (sum_pred*sum_gt).sum()

        return ((2.0 * intersection + smooth) / (sum_pred.sum() + sum_gt.sum() + smooth))
