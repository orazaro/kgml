#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Computer vision utils.
"""
from __future__ import division, print_function
import numpy as np
import cv2


def rgb_to_hsv(img, back=False):
    from_to = cv2.COLOR_HSV2RGB if back else cv2.COLOR_RGB2HSV
    if len(img.shape) == 3:
        return cv2.cvtColor(img, from_to)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        return cv2.cvtColor(img3, from_to).reshape(img_shape)


def hsv_to_rgb(img):
    return rgb_to_hsv(img, back=True)
