import os
import glob
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import argparse
import logging

# Configure logging to print information messages
logging.basicConfig(level=logging.INFO)

# DataFolder class to load images and masks
class SeepDataset(data.Dataset):
    def __init__(self, images_dir, masks_dir):
        """
        Initialize the dataset with the directory paths of images and masks.

        :param images_dir: Path to the images directory.
        :param masks_dir: Path to the masks directory.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = [os.path.basename(fn) for fn in glob.iglob(os.path.join(self.images_dir, '*.tif'))]

    def __getitem__(self, index):
        """
        Get the image and mask pair by index.

        :param index: Index of the image and mask.
        :return: A tuple of (image, mask, image_id).
        """
        image_id = self.image_filenames[index]

        # Load image
        image = Image.open(os.path.join(self.images_dir, image_id)).convert('I;16')
        image = np.array(image, dtype=np.float32)
        image = image / (2 ** 16 - 1)  # Normalize to range [0, 1]
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension

        # Load mask
        mask = Image.open(os.path.join(self.masks_dir, image_id)).convert('L')
        mask = np.array(mask, dtype=np.int64)
        mask = torch.from_numpy(mask).type(torch.LongTensor)

        return image, mask, image_id

    def __len__(self):
        """
        Get the length of the dataset.

        :return: Number of images in the dataset.
        """
        return len(self.image_filenames)


def create_dataloaders(train_images_dir, train_masks_dir, valid_images_dir, valid_masks_dir, eval_images_dir,
                       eval_masks_dir, train_batch_size, eval_batch_size):
    """
    Create DataLoader objects for training, validation, and evaluation datasets.

    :param train_images_dir: Directory for training images.
    :param train_masks_dir: Directory for training masks.
    :param valid_images_dir: Directory for validation images.
    :param valid_masks_dir: Directory for validation masks.
    :param eval_images_dir: Directory for evaluation images.
    :param eval_masks_dir: Directory for evaluation masks.
    :param train_batch_size: Batch size for training.
    :param eval_batch_size: Batch size for validation and evaluation.
    :return: DataLoader objects for training, validation, and evaluation datasets.
    """

    # Check if directories and images exist, log information
    if not os.path.exists(train_images_dir) or not os.path.exists(train_masks_dir):
        logging.error(f"Training directories not found: {train_images_dir} or {train_masks_dir}")
    if not os.path.exists(valid_images_dir) or not os.path.exists(valid_masks_dir):
        logging.error(f"Validation directories not found: {valid_images_dir} or {valid_masks_dir}")
    if not os.path.exists(eval_images_dir) or not os.path.exists(eval_masks_dir):
        logging.error(f"Evaluation directories not found: {eval_images_dir} or {eval_masks_dir}")

    train_loader = data.DataLoader(
        dataset=SeepDataset(train_images_dir, train_masks_dir),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = data.DataLoader(
        dataset=SeepDataset(valid_images_dir, valid_masks_dir),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )

    eval_loader = data.DataLoader(
        dataset=SeepDataset(eval_images_dir, eval_masks_dir),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=2
    )

    return train_loader, valid_loader, eval_loader


if __name__ == '__main__':
    # Argument parser for dynamic configuration
    parser = argparse.ArgumentParser(description='Seep Detection DataLoader Configuration')
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

    args = parser.parse_args()

    # Create data loaders
    train_loader, valid_loader, eval_loader = create_dataloaders(
        args.train_images_dir, args.train_masks_dir,
        args.valid_images_dir, args.valid_masks_dir,
        args.eval_images_dir, args.eval_masks_dir,
        args.train_batch_size, args.eval_batch_size
    )

    # Log dataset information
    logging.info(f"Training data: {len(train_loader.dataset)} images")
    logging.info(f"Validation data: {len(valid_loader.dataset)} images")
    logging.info(f"Evaluation data: {len(eval_loader.dataset)} images")
