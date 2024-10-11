import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import logging
import wandb  # Import Weights & Biases for logging and tracking
from loss_function import DiceLoss, IoULoss, FocalLoss, TverskyLoss
from model import UNet
from util import EarlyStopping, save_predictions, visualize_prediction, predict_and_save
from metric import dice_coefficient, precision_recall_f1, iou
from dataloader import create_dataloaders
import os

# Set seed for reproducibility
def set_seed(seed=12137):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Set seed to ensure reproducibility
set_seed()

# Setup logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Parse command line arguments
parser = argparse.ArgumentParser(description="UNet Model Training and Evaluation")
parser.add_argument('--train_images_dir', type=str, default='seep_detection/small_train_images_256/',
                    help='Directory path for training images.')
parser.add_argument('--train_masks_dir', type=str, default='seep_detection/small_train_masks_256/',
                    help='Directory path for training masks.')
parser.add_argument('--valid_images_dir', type=str, default='seep_detection/valid_images_256/',
                    help='Directory path for validation images.')
parser.add_argument('--valid_masks_dir', type=str, default='seep_detection/valid_masks_256/',
                    help='Directory path for validation masks.')
parser.add_argument('--eval_images_dir', type=str, default='seep_detection/eval_images_256/',
                    help='Directory path for evaluation images.')
parser.add_argument('--eval_masks_dir', type=str, default='seep_detection/eval_masks_256/',
                    help='Directory path for evaluation masks.')
parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training. Default is 16.')
parser.add_argument('--eval_batch_size', type=int, default=16,
                    help='Batch size for validation and evaluation. Default is 16.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience.')
parser.add_argument('--min_delta', type=float, default=0.001,
                    help='Minimum change to qualify as improvement for early stopping.')
parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss',
                    choices=['CrossEntropyLoss', 'DiceLoss', 'IoULoss', 'FocalLoss', 'TverskyLoss'],
                    help='Loss function to use.')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                    help='Choose to either train or evaluate the model.')
parser.add_argument('--project', type=str, default='unet-seep-detection', help='WandB project name')
parser.add_argument('--output_dir', type=str, default='output', help='Directory to save prediction outputs during evaluation')
args = parser.parse_args()

# Initialize WandB project and log hyperparameters
wandb.init(project=args.project, config={
    "epochs": args.epochs,
    "learning_rate": args.lr,
    "train_batch_size": args.train_batch_size,
    "eval_batch_size": args.eval_batch_size,
    "patience": args.patience,
    "min_delta": args.min_delta,
    "loss_function": args.loss_function
})

# Create data loaders for training, validation, and evaluation
train_loader, valid_loader, eval_loader = create_dataloaders(
    args.train_images_dir, args.train_masks_dir,
    args.valid_images_dir, args.valid_masks_dir,
    args.eval_images_dir, args.eval_masks_dir,
    args.train_batch_size, args.eval_batch_size
)

# Dictionary of loss functions to choose from
loss_functions = {
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "DiceLoss": DiceLoss(),
    "IoULoss": IoULoss(),
    "FocalLoss": FocalLoss(),
    "TverskyLoss": TverskyLoss()
}

# Set the loss function based on user input
criterion = loss_functions[args.loss_function]

# Model, optimizer, and EarlyStopping initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(input_channels=1, output_classes=8).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
es = EarlyStopping(patience=args.patience, min_delta=args.min_delta, mode='min', model=model, model_dir='trained_model', loss_fn = args.loss_function)


def train_one_epoch(epoch, model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    :param epoch: Current epoch number.
    :param model: The UNet model.
    :param dataloader: The DataLoader object for training data.
    :param optimizer: Optimizer for backpropagation.
    :param criterion: Loss function.
    :param device: The device (CPU or GPU).
    :return: Average training loss and metrics for the epoch.
    """
    model.train()
    train_loss = []
    metrics = {"dice": [], "iou": [], "precision": [], "recall": [], "f1": [], "accuracy": []}

    for batch_idx, (img, mask, _) in enumerate(dataloader):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

        # Calculate metrics
        dice = dice_coefficient(pred, mask)
        iou_score = iou(pred, mask)
        precision, recall, f1, accuracy = precision_recall_f1(pred, mask)

        metrics["dice"].append(dice)
        metrics["iou"].append(iou_score)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["accuracy"].append(accuracy)

    avg_train_loss = np.mean(train_loss)
    avg_metrics = {key: np.mean(val) for key, val in metrics.items()}

    log_message = f'[EPOCH {epoch}] Train Loss: {avg_train_loss:.4f} | Dice: {avg_metrics["dice"]:.4f} | IoU: {avg_metrics["iou"]:.4f} | Precision: {avg_metrics["precision"]:.4f} | Recall: {avg_metrics["recall"]:.4f} | F1: {avg_metrics["f1"]:.4f} | Accuracy: {avg_metrics["accuracy"]:.4f}'
    logging.info(log_message)
    print(log_message)

    # Log metrics to WandB
    wandb.log({
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'train_dice': avg_metrics['dice'],
        'train_iou': avg_metrics['iou'],
        'train_precision': avg_metrics['precision'],
        'train_recall': avg_metrics['recall'],
        'train_f1': avg_metrics['f1'],
        'train_accuracy': avg_metrics['accuracy']
    })

    return avg_train_loss, avg_metrics


def validate_one_epoch(epoch, model, dataloader, criterion, device):
    """
    Validates the model for one epoch.

    :param epoch: Current epoch number.
    :param model: The UNet model.
    :param dataloader: The DataLoader object for validation data.
    :param criterion: Loss function.
    :param device: The device (CPU or GPU).
    :return: Average validation loss and metrics for the epoch.
    """
    model.eval()
    valid_loss = []
    metrics = {"dice": [], "iou": [], "precision": [], "recall": [], "f1": [], "accuracy": []}

    with torch.no_grad():
        for batch_idx, (img, mask, _) in enumerate(dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            valid_loss.append(loss.item())

            # Calculate metrics
            dice = dice_coefficient(pred, mask)
            iou_score = iou(pred, mask)
            precision, recall, f1, accuracy = precision_recall_f1(pred, mask)

            metrics["dice"].append(dice)
            metrics["iou"].append(iou_score)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(f1)
            metrics["accuracy"].append(accuracy)

    avg_valid_loss = np.mean(valid_loss)
    avg_metrics = {key: np.mean(val) for key, val in metrics.items()}

    log_message = f'[EPOCH {epoch}] Valid Loss: {avg_valid_loss:.4f} | Dice: {avg_metrics["dice"]:.4f} | IoU: {avg_metrics["iou"]:.4f} | Precision: {avg_metrics["precision"]:.4f} | Recall: {avg_metrics["recall"]:.4f} | F1: {avg_metrics["f1"]:.4f} | Accuracy: {avg_metrics["accuracy"]:.4f}'
    logging.info(log_message)
    print(log_message)

    # Log metrics to WandB
    wandb.log({
        'epoch': epoch,
        'valid_loss': avg_valid_loss,
        'valid_dice': avg_metrics['dice'],
        'valid_iou': avg_metrics['iou'],
        'valid_precision': avg_metrics['precision'],
        'valid_recall': avg_metrics['recall'],
        'valid_f1': avg_metrics['f1'],
        'valid_accuracy': avg_metrics['accuracy']
    })

    return avg_valid_loss, avg_metrics


def train_model(model, train_loader, valid_loader, optimizer, criterion, es, epochs, device):
    """
    Trains the model for multiple epochs, applying early stopping when necessary.

    :param model: The UNet model.
    :param train_loader: DataLoader for training data.
    :param valid_loader: DataLoader for validation data.
    :param optimizer: Optimizer for backpropagation.
    :param criterion: Loss function.
    :param es: EarlyStopping object to monitor training.
    :param epochs: Number of epochs to train.
    :param device: The device (CPU or GPU).
    """
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train and validate
        train_loss, train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
        valid_loss, valid_metrics = validate_one_epoch(epoch, model, valid_loader, criterion, device)

        # Early stopping and saving best model
        es(valid_loss)
        if es.early_stop:
            print('Early stopping criterion met.')
            break
        elif es.counter == 0:
            es.save_checkpoint()
            print('Saving current best model.')

        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {elapsed_time:.2f} seconds')


def evaluate_model(model, dataloader, criterion, device, output_dir):
    """
    Evaluates the trained model on a test set and saves the predicted output along with plots.

    :param model: The trained UNet model.
    :param dataloader: DataLoader for evaluation data.
    :param criterion: Loss function.
    :param device: The device (CPU or GPU).
    :param output_dir: Directory to save the prediction results.
    :return: Final evaluation loss.
    """
    model.eval()
    eval_loss = []

    with torch.no_grad():
        for batch_idx, (img, mask, img_fns) in enumerate(dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            eval_loss.append(loss.item())

            pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
            
            # Save and visualize predictions
            save_predictions(pred_mask, img_fns, output_dir)
            visualize_prediction(img[0], mask[0], pred_mask[0], img_fns[0])

            print(f'[EVALUATE {batch_idx + 1}/{len(dataloader)}] Eval Loss: {loss.item():.4f}')

    avg_eval_loss = np.mean(eval_loss)
    logging.info(f'FINAL EVAL LOSS: {avg_eval_loss:.4f}')
    print(f'FINAL EVAL LOSS: {avg_eval_loss:.4f}')

    return avg_eval_loss


if __name__ == '__main__':
    # Start training or evaluation based on user input
    if args.mode == 'train':
        train_model(model, train_loader, valid_loader, optimizer, criterion, es, args.epochs, device)
    elif args.mode == 'eval':
        model = es.load_checkpoint()
        final_eval_loss = evaluate_model(model, eval_loader, criterion, device, args.output_dir)
