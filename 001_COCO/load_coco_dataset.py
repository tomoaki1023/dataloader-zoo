#!/usr/bin/env python
# coding:utf-8

import glob
import os
import sys

import numpy as np
import simplejson as json
from tqdm import tqdm


def load_dataset(dataset_param):
    train_data_paths, train_label_data = format_data_and_labels(dataset_param, "train")
    val_data_paths, val_label_data = format_data_and_labels(dataset_param, "val")

    return train_data_paths, train_label_data, val_data_paths, val_label_data


def format_data_and_labels(dataset_param, subset):
    format_data = []
    format_labels = []

    image_paths = get_image_paths(dataset_param[f"{subset}_image_directory"])
    coco_labels = get_coco_label_data(dataset_param[f"{subset}_label_file_path"], subset)

    assert len(image_paths) == len(coco_labels)

    for image_path in image_paths:
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])

        if image_id not in coco_labels.keys():
            continue

        if len(coco_labels[image_id]["objects"]) == 0:
            continue

        if dataset_param["delete_iscrowd_image"] and check_iscrowd_object(coco_labels[image_id]["objects"]):
            continue

        if not os.path.isfile(image_path):
            print(f"The file '{image_path}' does not exist.")
            sys.exit(1)

        format_data.append(image_path)
        format_labels.append(format_label(coco_labels[image_id]["objects"]))

    assert len(format_data) == len(format_labels)

    return format_data, format_labels


def get_image_paths(dir_path):
    # collect paths of existing jpg images
    image_paths = [
        path
        for path in glob.glob(os.path.join(dir_path, "*"))
        if path.lower().endswith(".jpg") and os.path.isfile(path)
    ]
    return image_paths


def get_coco_label_data(label_file_path, subset):
    if not os.path.isfile(label_file_path):
        print(f"'{label_file_path}' does not exist.")
        sys.exit(1)

    # coco_json_data: dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
    coco_json_data = read_json_file(label_file_path, subset)

    coco_labels = create_labels(coco_json_data)

    return coco_labels


def read_json_file(label_file_path, subset):
    try:
        with open(label_file_path, "r", encoding="utf-8") as f:
            coco_json_data = json.load(f, object_hook=lambda obj: print_load_progress(obj, subset))

            return coco_json_data
    except Exception as e:
        print(e)
        sys.exit(1)


def print_load_progress(obj, subset):
    images_dict_info = obj.get("images")
    if images_dict_info:
        for _ in tqdm(images_dict_info, desc=f"[Loading MSCOCO Dataset: {subset} json file]"):
            pass

    return obj


def create_labels(coco_json_data):
    coco_labels = {
        image_info["id"]: {"file_name": image_info["file_name"], "objects": []}
        for image_info in coco_json_data["images"]
    }

    for annotation_info in coco_json_data["annotations"]:
        if annotation_info["image_id"] not in coco_labels:
            continue

        object_dict = {
            "bbox": annotation_info["bbox"],
            "iscrowd": annotation_info["iscrowd"],
            "category_id": annotation_info["category_id"],
        }

        coco_labels[annotation_info["image_id"]]["objects"].append(object_dict)

    return coco_labels


def check_iscrowd_object(objects):
    for object_info in objects:
        if object_info["iscrowd"] == 1:
            return True

    return False


def format_label(objects):
    label_info = []
    for object_info in objects:
        bbox = object_info["bbox"]

        bbox_cx = bbox[0] + (bbox[2] * 0.5)
        bbox_cy = bbox[1] + (bbox[3] * 0.5)

        # [cx, cy, width, height, category_id]
        label_info.append([bbox_cx, bbox_cy, bbox[2], bbox[3], object_info["category_id"]])

    return np.array(label_info, dtype=np.float32)
