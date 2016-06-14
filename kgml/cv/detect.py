#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev
# License:  BSD 3 clause
"""
    Object Detector
"""
from __future__ import division, print_function
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from scipy import ndimage


def transform_hs(img, hue_min=0.45, hue_max=0.60, satur_min=0.4):
    """ Transform image using hue and saturation features """
    # select using hue
    hue = rgb_to_hsv(img)[:, :, 0]
    img = img.copy()
    img[hue < hue_min] = 0
    img[hue > hue_max] = 0

    # select usin saturation
    img = rgb_to_hsv(img)[:, :, 1]
    binary_img = img > satur_min

    # Remove small white regions
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)

    return close_img


class HueSaturationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y, shape=None):
        pass

    def transform(self, X):
        pass


def test_transform_hs():
    shape = (32, 64, 3)
    img = np.full(shape, 0.5)
    plt.imshow(img, cmap='gray')
    plt.show()

if __name__=='__main__':
    test_transform_hs()
