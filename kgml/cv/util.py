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


def rgb_to_hsv(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        img_shape = img.shape
        img3 = img[np.newaxis, :, :]
        return cv2.cvtColor(img3, cv2.COLOR_RGB2HSV).reshape(img_shape)
