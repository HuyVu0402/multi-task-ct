import torch
import torch.nn as nn
import torch.nn.functional as F


# ============== SEGMENTATION LOSSES ==============

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred, target: [B,1,H,W], pred là prob 0-1
        """
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        denom = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2. * intersection + self.smooth) / (denom + self.smooth)
        loss = 1 - dice.mean()
        return loss


class CombinedSegLoss(nn.Module):
    """Dice + BCE cho segmentation"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.dw = dice_weight
        self.bw = bce_weight

    def forward(self, pred, target):
        return self.dw * self.dice(pred, target) + self.bw * self.bce(pred, target)


# ============== MULTI-TASK LOSS ==============

class MultiTaskLoss(nn.Module):
    """
    loss = w_seg * (Dice + BCE) + w_cls * CrossEntropy
    """
    def __init__(self, w_seg=1.0, w_cls=1.0, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.seg_loss = CombinedSegLoss(dice_weight=dice_weight, bce_weight=bce_weight)
        self.cls_loss = nn.CrossEntropyLoss()
        self.w_seg = w_seg
        self.w_cls = w_cls

    def forward(self, pred_seg, gt_seg, pred_cls, gt_cls):
        """
        pred_seg: [B,1,H,W]
        gt_seg  : [B,1,H,W]
        pred_cls: [B,2] (logits)
        gt_cls  : [B] (long)
        """
        loss_seg = self.seg_loss(pred_seg, gt_seg)
        loss_cls = self.cls_loss(pred_cls, gt_cls)
        loss = self.w_seg * loss_seg + self.w_cls * loss_cls
        return loss, loss_seg.detach(), loss_cls.detach()
