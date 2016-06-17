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
    if len(img.shape) == 3:
        img2 = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        img2 = cv2.cvtColor(img3, cv2.COLOR_RGB2HSV).reshape(img_shape)
    return img_norm(img2, is_hsv=True)


def hsv_to_rgb(img):
    img = img_norm(img, is_hsv=True, back=True)
    if len(img.shape) == 3:
        img2 = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        img2 = cv2.cvtColor(img3, cv2.COLOR_HSV2RGB).reshape(img_shape)
    return img2


def test_rgb_to_hsv():
    shape = (2, 2, 3)
    img = np.full(shape, 0.5)
    img[:, :1, 0] = 0.10
    img[:, 1:, 0] = 0.90
    img = hsv_to_rgb(img)
    print("rgb:\n", img)
    img_hsv = rgb_to_hsv(img)
    print("hsv:\n", img_hsv)
    img2 = hsv_to_rgb(img_hsv)
    print("rgb2:\n", img2)
    
    # plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    test_rgb_to_hsv()
