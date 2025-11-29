import torch


def dice_coefficient(pred, target, threshold=0.5):
    """
    pred, target: [B,1,H,W], prob 0-1
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3))

    dice = (2 * intersection + 1e-7) / (union + 1e-7)
    return dice.mean().item()


def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean().item()


def classification_metrics(logits, labels):
    """
    logits: [B,2], labels: [B]
    Trả về accuracy (float)
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().mean().item()
    return correct
