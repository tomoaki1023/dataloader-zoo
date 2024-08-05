#!/usr/bin/env python
# coding:utf-8

import numpy as np
from PIL import Image
from utils import imread


class COCODataset(object):
    def __init__(self, image_list, labels, input_image_size, train):
        self.image_list = image_list
        self.labels = labels
        self.input_height, self.input_width = input_image_size
        self.train = train

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path, label = self.image_list[idx], self.labels[idx]
        image = imread(image_path)
        image, label = self.preprocess_data(image, label)
        return image, label

    def preprocess_data(self, image, label):
        # TODO: Add your custom preprocessing steps here
        # This method should implement any data augmentation, normalization,
        # or other preprocessing steps required for your specific task.
        #
        # Example preprocessing steps might include:
        # - Resizing the image
        # - Normalizing pixel values
        # - Data augmentation (for training set)
        # - Converting labels to the required format
        #
        # Remember to handle both 'image' and 'label' as needed for your task.

        image, label = self.scale_image_with_aspect_ratio(image, label)

        return image, label

    def scale_image_with_aspect_ratio(self, src_image, src_label):
        """
        Scale the image while maintaining the aspect ratio.

        Args:
            src_image (PIL.Image): Image data to scale.
            src_label (np.ndarray): Label data.

        Returns:
            tuple: Tuple containing the scaled image data and updated label.
        """

        # Initialize
        dst_label = np.zeros((src_label.shape[0], src_label.shape[1]), dtype=src_label.dtype)
        dst_image = Image.new("RGB", (self.input_width, self.input_height), color=(0, 0, 0))

        # Resize image
        resize_width_rate = self.input_width / src_image.height
        resize_height_rate = self.input_height / src_image.width
        resize_rate = min(resize_width_rate, resize_height_rate)
        resize_width = int(src_image.width * resize_rate)
        resize_height = int(src_image.height * resize_rate)

        # Image downscaling
        if (self.input_width * self.input_height) < (src_image.width * src_image.height):
            resized_image = src_image.resize((resize_width, resize_height), resample=Image.LANCZOS)
        # Image upscaling
        else:
            resized_image = src_image.resize((resize_width, resize_height), resample=Image.BICUBIC)

        # Pasting the resized image onto a black image
        dst_dx, dst_dy = self.paste_image_at_random_position(resized_image, dst_image)

        # Updating object coordinates
        dst_label[:, 0] = (src_label[:, 0] * resize_rate) + dst_dx
        dst_label[:, 1] = (src_label[:, 1] * resize_rate) + dst_dy
        dst_label[:, 2] = src_label[:, 2] * resize_rate
        dst_label[:, 3] = src_label[:, 3] * resize_rate
        dst_label[:, 4] = src_label[:, 4]

        return np.array(dst_image, dtype=np.float32), dst_label

    def paste_image_at_random_position(self, src_image, dst_image):
        """
         Paste the source image onto the destination image at a random position.

         Args:
            src_image (PIL.Image): Source image to paste.
            dst_image (PIL.Image): Destination image to paste onto.

        Returns:
            tuple: Tuple containing the pasted coordinates (x, y).
        """

        if self.train:
            pb = np.random.rand()
        else:
            pb = 0

        # Calculate the position to paste the image
        dst_dx = int(pb * (dst_image.size[0] - src_image.size[0]))
        dst_dy = int(pb * (dst_image.size[1] - src_image.size[1]))

        # Set image
        dst_image.paste(src_image, (dst_dx, dst_dy))

        return dst_dx, dst_dy
