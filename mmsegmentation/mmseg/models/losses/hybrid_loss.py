import torch
import torch.nn as nn

import numpy
import torch.nn.functional as F

from segmentation_models_pytorch.losses import TverskyLoss, JaccardLoss, DiceLoss

from ..builder import LOSSES
from .utils import weighted_loss

"""
class FocalTverskyLoss(nn.Module):
    def __init__(self, focal_alpha, focal_beta, focal_gamma, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = focal_alpha
        self.beta = focal_beta
        self.gamma = focal_gamma

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(targets.long(), 0, num_classes - 1),
            num_classes=num_classes)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = one_hot_target.view(-1)

        # True Positives, False Positives & False Negatives
        TP = torch.sum(inputs * targets)
        FP = torch.sum((1 - targets) * inputs)
        FN = torch.sum(targets * (1 - inputs))

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        FocalTversky = (1 - Tversky).pow(self.gamma)

        return FocalTversky


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer

        inputs = F.softmax(inputs, dim=1)
        num_classes = inputs.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(targets.long(), 0, num_classes - 1),
            num_classes=num_classes)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = one_hot_target.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU
"""

@LOSSES.register_module
class HybridLoss(nn.Module):

    def __init__(self, focal_alpha=0.5, focal_beta=0.4, focal_gamma=4/3, reduction='mean', loss_weight=1.0):
        super(HybridLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.focal_alpha = focal_alpha
        self.focal_beta = focal_beta
        self.focal_gamma = focal_gamma
        self._loss_name = 'loss_hybrid'

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None, **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        """
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)
        """

        #pred = torch.argmax(pred, dim=-1)

        pred = F.softmax(pred, dim=1)
        #target = F.softmax(target, dim=1)

        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)

        loss = 0


        #dice_loss = DiceLoss(mode='binary')
        focal_tversky_loss = TverskyLoss(mode='binary', alpha=self.focal_alpha, beta=self.focal_beta, gamma=self.focal_gamma)
        iou_loss = JaccardLoss(mode='binary')
        loss = 0
        for i in range(num_classes):
            a = pred[:, i]
            b = one_hot_target[..., i]
            #loss += self.loss_weight*dice_loss(pred[:, i], one_hot_target[..., i])
            loss += self.loss_weight * (focal_tversky_loss(a, b) + iou_loss(a, b))
        return loss/num_classes

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name