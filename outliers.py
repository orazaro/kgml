#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    outliers detection
"""
import numpy as np
from sklearn.covariance import EllipticEnvelope

def search_outliers_EllipticEnvelope(X):
    clf = EllipticEnvelope(contamination=0.2)
    clf.fit(X)
    is_outliers = clf.predict(X)
    return is_outliers

def search_outliers_array(data, m = 6.):
    return abs(data - np.mean(data)) > m * np.std(data)

def search_outliers_array2(data, m = 6.):
    """ http://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list """
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0
    return s>m

def search_outliers(X, m = 6., verbose=1):
    nrows,ncols = X.shape
    outliers = np.array([0.0] * nrows)
    for j in range(ncols):
        isout = search_outliers_array(X[:,j],m)
        if np.any(isout):
            bad = np.where(isout)[0]
            outliers[bad] += 1.0
            if verbose>1:
                print("outliers col:%d row_vals:%r"%(j,zip(bad,X[bad,j]))),
                print "data: ",np.mean(X[:,j]),"+-",np.std(X[:,j])
    if verbose>0:
        print "outliers:",outliers[outliers>m*np.std(outliers)]
    return outliers == 0


def test():
    data = np.arange(100)/100.
    data[50] = 3.
    print np.any(search_outliers_array(data))
    X = np.array([range(50),range(50),range(50)])/50.
    X[1,10] = 3.
    X[1,14] = 15.
    X[0,1] = 33.
    #print search_outliers(X)
    good = search_outliers(X.T)
    print "good rows:",sum(good)
    #print X.T[good,:]

if __name__ == "__main__":
    test()
