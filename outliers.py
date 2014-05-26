#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    outliers detection
"""
import numpy as np
from sklearn.covariance import EllipticEnvelope

def search_outliers(X):
    clf = EllipticEnvelope(contamination=0.1)
    clf.fit(X)
    is_outliers = clr.predict(X)
    return is_outliers

