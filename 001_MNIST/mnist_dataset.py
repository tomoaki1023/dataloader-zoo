#!/usr/bin/env python
# coding:utf-8

import numpy as np


class MNISTDataset(object):
    def __init__(self, images, labels, train):
        self.images = images
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
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
        return image, label
