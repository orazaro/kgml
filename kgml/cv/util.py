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


def img_norm(img, is_hsv=False):
    divs = np.array([255., 255., 255.])
    if is_hsv:
        divs[0] = 179.
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] = img[:, :, i] / divs[i]
    return img


def rgb_to_hsv(img, back=False):
    from_to = cv2.COLOR_HSV2RGB if back else cv2.COLOR_RGB2HSV
    if len(img.shape) == 3:
        img2 = cv2.cvtColor(img, from_to)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        img2 = cv2.cvtColor(img3, from_to).reshape(img_shape)
    
    return img_norm(img2, is_hsv=True)


def hsv_to_rgb(img):
    return rgb_to_hsv(img, back=True)


def test_rgb_to_hsv():
    img = np.zeros((1, 2, 3), dtype=np.float32)
    img[:, :] = np.array([10, 30, 150], dtype=np.float32)
    img = img / 255.
    print(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print(img_hsv)
    print(rgb_to_hsv(img))
    
    # plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    test_rgb_to_hsv()
