import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Loss Definition
class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6):
        if pred.size(1) > 1:
            pred = torch.softmax(pred, dim=1)
            target = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        else:
            pred = torch.sigmoid(pred)
            target = target.float()

        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()

# IoU Loss Definition
class IoULoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6):
        if pred.size(1) > 1:
            pred = torch.softmax(pred, dim=1)
            target = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        else:
            pred = torch.sigmoid(pred)
            target = target.float()

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou.mean()

# Focal Loss Definition
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        BCE_loss = F.cross_entropy(pred, target)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss

# Tversky Loss Definition
class TverskyLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6, alpha=0.5, beta=0.5):
        if pred.size(1) > 1:
            pred = torch.softmax(pred, dim=1)
            target = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        else:
            pred = torch.sigmoid(pred)
            target = target.float()

        TP = (pred * target).sum(dim=(2, 3))
        FP = ((1 - target) * pred).sum(dim=(2, 3))
        FN = (target * (1 - pred)).sum(dim=(2, 3))

        tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1 - tversky_index.mean()

# Testing each loss function with dummy data
if __name__ == "__main__":
    # Create dummy prediction and target tensors for testing
    pred = torch.randn(4, 2, 256, 256)  # Batch size 4, 2 classes, 256x256 images
    target = torch.randint(0, 2, (4, 256, 256))  # Random binary mask

    # Test DiceLoss
    dice_loss = DiceLoss()
    dice_loss_value = dice_loss(pred, target)
    print(f"Dice Loss: {dice_loss_value.item()}")

    # Test IoULoss
    iou_loss = IoULoss()
    iou_loss_value = iou_loss(pred, target)
    print(f"IoU Loss: {iou_loss_value.item()}")

    # Test FocalLoss
    focal_loss = FocalLoss(alpha=1, gamma=2)
    focal_loss_value = focal_loss(pred, target)
    print(f"Focal Loss: {focal_loss_value.item()}")

    # Test TverskyLoss
    tversky_loss = TverskyLoss()
    tversky_loss_value = tversky_loss(pred, target)
    print(f"Tversky Loss: {tversky_loss_value.item()}")
