#!/usr/bin/env python
# coding:utf-8

import argparse
import sys

import numpy as np
import torch
import yaml
from coco_dataloader import DataLoader
from coco_dataset import COCODataset
from load_coco_dataset import load_dataset


def read_config_file(config_file_path):
    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    except Exception as e:
        print(e)
        sys.exit(1)


def collate_np_to_torch(images: np.array, targets: list):
    images = torch.from_numpy(images).to(dtype=torch.float32)
    targets = [torch.from_numpy(target).to(dtype=torch.float32) for target in targets]
    return images, targets


def main(args):
    epochs = 5
    batch_size = 10

    input_image_size = (640, 640)

    dataset_param = read_config_file(args.config)

    train_data, train_labels, val_data, val_labels = load_dataset(dataset_param)

    train_dataset = COCODataset(train_data, train_labels, input_image_size, train=True)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
        print(f"epoch: {epoch}")

        for iteration, (images, targets) in enumerate(train_dataloader, 1):
            print(f"iteration: {iteration}")
            print("images: ", images.shape)

            for i, target in enumerate(targets):
                print(f"target of image[{i}]:", target.shape)

            # images, targets = collate_np_to_torch(images, targets)
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config file path", default="config.yml")
    args = parser.parse_args()

    main(args)
