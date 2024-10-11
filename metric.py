import torch

# Dice Coefficient Metric
def dice_coefficient(pred, target, smooth=1):
    """
    Compute the Dice Coefficient, a measure of overlap between predicted and target masks.
    
    :param pred: Predicted mask tensor (N, C, H, W) where C is the number of classes.
    :param target: Ground truth mask tensor (N, H, W) with integer class labels.
    :param smooth: Smoothing term to avoid division by zero (default is 1).
    :return: Dice coefficient averaged over the batch.
    """
    # Convert predictions to class indices by taking argmax across channels (for multi-class tasks)
    pred = torch.argmax(pred, dim=1)
    
    # Add a channel dimension to both pred and target (necessary for broadcasting)
    pred = pred.unsqueeze(1)
    pred = pred.float()  # Convert to float for calculations
    target = target.unsqueeze(1).float()  # Convert target to float and add channel dimension
    
    # Calculate intersection and union of predicted and target masks
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    
    # Return the mean Dice coefficient across the batch
    return dice.mean().item()


# IoU (Intersection over Union) Metric
def iou(pred, target, smooth=1):
    """
    Compute the IoU (Intersection over Union), also known as the Jaccard Index, for segmentation tasks.
    
    :param pred: Predicted mask tensor (N, C, H, W) where C is the number of classes.
    :param target: Ground truth mask tensor (N, H, W) with integer class labels.
    :param smooth: Smoothing term to avoid division by zero (default is 1).
    :return: IoU averaged over the batch.
    """
    # Convert predictions to class indices by taking argmax across channels (for multi-class tasks)
    pred = torch.argmax(pred, dim=1)
    
    # Add a channel dimension to both pred and target (necessary for broadcasting)
    pred = pred.unsqueeze(1)
    pred = pred.float()  # Convert to float for calculations
    target = target.unsqueeze(1).float()  # Convert target to float and add channel dimension
    
    # Calculate intersection and union of predicted and target masks
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    # IoU calculation
    iou = (intersection + smooth) / (union + smooth)
    
    # Return the mean IoU across the batch
    return iou.mean().item()


# Precision, Recall, F1, and Accuracy Metrics
def precision_recall_f1(pred, target, smooth=1):
    """
    Compute Precision, Recall, F1-Score, and Accuracy for segmentation tasks.
    
    :param pred: Predicted mask tensor (N, C, H, W) where C is the number of classes.
    :param target: Ground truth mask tensor (N, H, W) with integer class labels.
    :param smooth: Smoothing term to avoid division by zero (default is 1).
    :return: Precision, Recall, F1-Score, and Accuracy averaged over the batch.
    """
    # Convert predictions to class indices by taking argmax across channels (for multi-class tasks)
    pred = torch.argmax(pred, dim=1)
    
    # Add a channel dimension to both pred and target (necessary for broadcasting)
    pred = pred.unsqueeze(1)
    pred = pred.float()  # Convert to float for calculations
    target = target.unsqueeze(1).float()  # Convert target to float and add channel dimension
    
    # Calculate True Positives, Precision, and Recall
    true_positive = (pred * target).sum(dim=(2, 3))
    precision = (true_positive + smooth) / (pred.sum(dim=(2, 3)) + smooth)
    recall = (true_positive + smooth) / (target.sum(dim=(2, 3)) + smooth)
    
    # F1 Score calculation
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    # Calculate Accuracy
    correct_predictions = (pred == target).float().sum(dim=(2, 3))
    total_pixels = target.numel()
    accuracy = correct_predictions.sum() / total_pixels
    
    # Return Precision, Recall, F1-Score, and Accuracy
    return precision.mean().item(), recall.mean().item(), f1.mean().item(), accuracy.item()


# Testing each metric function with dummy tensors
if __name__ == '__main__':
    # Create dummy prediction tensor with random values (4 samples, 2 classes, 256x256 images)
    pred = torch.randn(4, 2, 256, 256)
    
    # Create dummy target tensor with random class labels (binary for simplicity)
    target = torch.randint(0, 2, (4, 256, 256))
    
    # Test Dice Coefficient
    dice_value = dice_coefficient(pred, target)
    print(f"Dice Coefficient: {dice_value}")
    
    # Test IoU
    iou_value = iou(pred, target)
    print(f"IoU: {iou_value}")
    
    # Test Precision, Recall, F1, and Accuracy
    precision, recall, f1, accuracy = precision_recall_f1(pred, target)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Accuracy: {accuracy}")
