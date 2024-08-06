#!/usr/bin/env python
# coding:utf-8

import gzip

import numpy as np


def load_dataset(dataset_param):
    train_images_path = dataset_param["train_images_file_path"]
    train_labels_path = dataset_param["train_labels_file_path"]
    val_images_path = dataset_param["val_images_file_path"]
    val_labels_path = dataset_param["val_labels_file_path"]

    with gzip.open(train_labels_path, "rb") as f:
        train_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(train_images_path, "rb") as f:
        train_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    with gzip.open(val_labels_path, "rb") as f:
        val_labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with gzip.open(val_images_path, "rb") as f:
        val_images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(val_labels), 28, 28)

    return train_images, train_labels, val_images, val_labels
