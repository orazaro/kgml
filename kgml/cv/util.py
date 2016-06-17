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


def rgb_to_hsv(img, back=False):
    from_to = cv2.COLOR_HSV2RGB if back else cv2.COLOR_RGB2HSV
    if len(img.shape) == 3:
        img2 = cv2.cvtColor(img, from_to)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        img2 = cv2.cvtColor(img3, from_to).reshape(img_shape)
    
    divs = (179., 255., 255.)
    img2 = img2.astype(np.float32)
    for i in range(3):
        img2[:, :, i] = img2[:, :, i] / divs[i]
    return img2


def hsv_to_rgb(img):
    return rgb_to_hsv(img, back=True)


def test_rgb_to_hsv():
    img = np.zeros((1, 2, 3), dtype=np.float32)
    img[:, :] = [10, 30, 150]
    img = img / 255.
    print(img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print(img_hsv[0, 0, 0])
    
    # plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    test_rgb_to_hsv()
