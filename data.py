import os
import shutil
import glob
import random
import argparse
import logging

# Configure logging to print information messages
logging.basicConfig(level=logging.INFO)

class DataSplitter:
    def __init__(self, base_path, train_split=0.8, valid_split=0.1, ext='tif'):
        """
        Initialize the data splitter with paths, split ratios, and file extension.

        :param base_path: Base path to the dataset.
        :param train_split: Proportion of the data to use for training.
        :param valid_split: Proportion of the data to use for validation.
        :param ext: File extension of the images (default: 'tif').
        """
        self.base_path = base_path
        self.train_split = train_split
        self.valid_split = valid_split
        self.ext = ext
        self.all_images_path = os.path.join(base_path, 'train_images_256')
        self.all_masks_path = os.path.join(base_path, 'train_masks_256')

        self.train_images_path = os.path.join(base_path, 'small_train_images_256')
        self.train_masks_path = os.path.join(base_path, 'small_train_masks_256')
        self.valid_images_path = os.path.join(base_path, 'valid_images_256')
        self.valid_masks_path = os.path.join(base_path, 'valid_masks_256')
        self.eval_images_path = os.path.join(base_path, 'eval_images_256')
        self.eval_masks_path = os.path.join(base_path, 'eval_masks_256')

        # Create the necessary directories for the splits
        self._create_directories()

    def _create_directories(self):
        """Create directories for train, validation, and evaluation sets."""
        os.makedirs(self.train_images_path, exist_ok=True)
        os.makedirs(self.train_masks_path, exist_ok=True)
        os.makedirs(self.valid_images_path, exist_ok=True)
        os.makedirs(self.valid_masks_path, exist_ok=True)
        os.makedirs(self.eval_images_path, exist_ok=True)
        os.makedirs(self.eval_masks_path, exist_ok=True)

    def split_data(self):
        """
        Split the dataset into training, validation, and evaluation sets based on defined ratios.
        """
        # Get all image files with the specified extension and shuffle them
        all_image_files = sorted(glob.glob(os.path.join(self.all_images_path, f'*.{self.ext}')))
        random.shuffle(all_image_files)

        # Calculate the number of files for each split
        total_files = len(all_image_files)
        train_split = int(self.train_split * total_files)
        valid_split = int(self.valid_split * total_files)

        # Split the files into training, validation, and evaluation sets
        train_files = all_image_files[:train_split]
        valid_files = all_image_files[train_split:train_split + valid_split]
        eval_files = all_image_files[train_split + valid_split:]

        # Copy the files to their respective directories
        self._copy_files(train_files, self.train_images_path, self.train_masks_path)
        self._copy_files(valid_files, self.valid_images_path, self.valid_masks_path)
        self._copy_files(eval_files, self.eval_images_path, self.eval_masks_path)

        # Verify the split result by logging the number of files
        self._verify_split()

    def _copy_files(self, file_list, target_image_dir, target_mask_dir):
        """
        Copy the image and corresponding mask files to the target directories.

        :param file_list: List of image file paths to copy.
        :param target_image_dir: Directory to store the copied images.
        :param target_mask_dir: Directory to store the copied masks.
        """
        for file_path in file_list:
            file_name = os.path.basename(file_path)
            mask_file_path = os.path.join(self.all_masks_path, file_name)

            # Check if the mask file exists before copying
            if not os.path.exists(mask_file_path):
                logging.warning(f"Mask file missing for image {file_name}, skipping.")
                continue

            # Check if the file already exists in the target directory
            target_image_file = os.path.join(target_image_dir, file_name)
            target_mask_file = os.path.join(target_mask_dir, file_name)

            if os.path.exists(target_image_file) and os.path.exists(target_mask_file):
                logging.info(f"Files for {file_name} already exist, skipping copy.")
                continue  # Skip if files already exist

            # Copy the image and mask to the target directory
            shutil.copy(file_path, target_image_file)
            shutil.copy(mask_file_path, target_mask_file)

    def _verify_split(self):
        """Log the number of images in each set to verify the split."""
        logging.info(f"Total images: {len(os.listdir(self.all_images_path))}")
        logging.info(f"Training images: {len(os.listdir(self.train_images_path))}")
        logging.info(f"Validation images: {len(os.listdir(self.valid_images_path))}")
        logging.info(f"Evaluation images: {len(os.listdir(self.eval_images_path))}")


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Data Splitter for Oil Seep Detection')
    parser.add_argument('--base_path', type=str, default='seep_detection/',
                        help='Base path to the dataset. Default is "seep_detection/".')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Proportion of the data to use for training. Default is 0.8.')
    parser.add_argument('--valid_split', type=float, default=0.1,
                        help='Proportion of the data to use for validation. Default is 0.1.')
    parser.add_argument('--ext', type=str, default='tif',
                        help='File extension of the images. Default is "tif".')

    args = parser.parse_args()

    # Initialize the DataSplitter with user-provided or default arguments
    splitter = DataSplitter(base_path=args.base_path, train_split=args.train_split, valid_split=args.valid_split, ext=args.ext)
    splitter.split_data()
