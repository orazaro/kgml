#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Computer vision utils.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2


def img_norm(img, is_hsv=False, back=False):
    """Normalize from int8 to [0,1] and back.""" 
    divs = np.array([255., 255., 255.])
    if is_hsv:
        divs[0] = 179.
    if back:
        for i in range(3):
            img[:, :, i] = img[:, :, i] * divs[i]
        return img.astype(np.uint8)
    else:
        img = img.astype(np.float32)
        for i in range(3):
            img[:, :, i] = img[:, :, i] / divs[i]
        return img


def rgb_to_hsv(img):
    """RGB to HSV using opencv."""
    if len(img.shape) == 3:
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img2 = img_norm(img2, is_hsv=True)
    else:
        img3 = img[np.newaxis, :, :]
        img2 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV)
        img2 = img_norm(img2, is_hsv=True)[0, :, :]
    return img2


def hsv_to_rgb(img):
    """HSV to RGB using opencv."""
    if len(img.shape) == 3:
        img = img_norm(img, is_hsv=True, back=True)
        img2 = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    else:
        img3 = img[np.newaxis, :, :]
        img3 = img_norm(img3, is_hsv=True, back=True)
        img2 = cv2.cvtColor(img3, cv2.COLOR_HSV2RGB)[0, :, :]
    return img2


def test_rgb_to_hsv():
    shape = (3, 2, 3)
    img = np.full(shape, 0.5)
    img[:, :1, 0] = 0.10
    img[:, 1:, 0] = 0.90
    img = hsv_to_rgb(img)
    print("rgb:\n", img)
    img_hsv = rgb_to_hsv(img)
    print("hsv:\n", img_hsv)
    img2 = hsv_to_rgb(img_hsv)
    print("rgb2:\n", img2)

    if True:
        fig, axarr = plt.subplots(1, 3)
        axarr[0].imshow(img)
        axarr[1].imshow(img_hsv)
        axarr[2].imshow(img2)
        plt.show()

if __name__ == '__main__':
    test_rgb_to_hsv()
