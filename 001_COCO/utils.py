#!/usr/bin/env python
# coding:utf-8

import numpy as np
from PIL import Image


def imread(image_path: str) -> np.array:
    """
    Read an image file using Pillow and return it as a 3-channel numpy array in RGB order.
    :param image_path: Path to the image file to read
    :return: Numpy array of the read image
    """
    try:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            np_image = np.array(rgb_image)
        return np_image

    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}. Error: {str(e)}")
