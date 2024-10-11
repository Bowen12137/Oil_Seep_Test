import time
import zipfile
import os
import glob
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import torch.nn.functional as F
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min', model=None, model_dir='model',loss_fn = "CrossEntropyLoss"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.model = model
        self.model_dir = model_dir
        self.name_loss = loss_fn

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint()
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        elif self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            raise ValueError("Mode should be 'min' or 'max'")

    def save_checkpoint(self):
        if not os.path.exists(self.model_dir):
            print(f'Creating model directory: {self.model_dir}')
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), f'{self.model_dir}/{self.name_loss}_unet.pth')
        print(f'Saved best model to {self.model_dir}/{self.name_loss}_unet.pth')

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(f'{self.model_dir}/{self.name_loss}_unet.pth'))
        return self.model

def save_predictions(predictions, filenames, output_directory):
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        print(f'Creating output directory: {output_directory}')
        os.makedirs(output_directory)

    # Iterate through each prediction and save it as an image file
    for idx, prediction in enumerate(predictions):
        prediction = torch.squeeze(prediction, dim=0)
        prediction_np = prediction.cpu().numpy().astype(np.uint8)  # Ensure the array is in uint8 format
        prediction_img = Image.fromarray(prediction_np)
        prediction_img.save(os.path.join(output_directory, filenames[idx]))




def visualize_prediction(img, mask, pred_mask, img_id):
    img = img.cpu().numpy().squeeze()
    mask = mask.cpu().numpy().squeeze()
    pred_mask = pred_mask.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction Mask')
    axes[2].axis('off')

    plt.suptitle(f'Image ID: {img_id}')
    plt.show()

def predict_and_save(model, dataloader, criterion, device, output_dir):
    model.eval()
    all_loss = []

    with torch.no_grad():
        for batch_idx, (img, mask, img_fns) in enumerate(dataloader):
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            loss = criterion(pred, mask)
            all_loss.append(loss.item())

            pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
            pred_mask = torch.chunk(pred_mask, chunks=16, dim=0)
            save_predictions(pred_mask, img_fns, output_dir)

            visualize_prediction(img[0], mask[0], pred_mask[0], img_fns[0])

            print(f'[PREDICT {batch_idx + 1}/{len(dataloader)}] Loss: {loss.item():.4f}')

    avg_loss = np.mean(all_loss)
    print(f'FINAL PREDICT LOSS: {avg_loss:.4f}')
    return avg_loss
