#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    features selection:
    remove market mode
    http://www.santafe.edu/media/workingpapers/07-08-030.pdf (19)
"""
import numpy as np
from scipy import stats
from feasel import VarSel

def rmm(X, verbose=0):
    """ Remove market mode and calculate B,A and D matrices:
        X_ij = B_j + A_j * M_i + D_ij
        M_i = <X>i
        for individual ids
    """
    if True:
        M_i = np.average(X,axis=1).ravel()
    else:
        from sklearn import preprocessing, decomposition
        X1 = VarSel(k=9000).fit_transform(X)
        X1 = preprocessing.StandardScaler(with_mean=True).fit_transform(X1)
        X2 = decomposition.RandomizedPCA(n_components=5, whiten=True,
            random_state=1).fit_transform(X1)
        M_i = X2[:,0].ravel()
    if verbose > 1:
        print "X:",X
        print "M_i: %r +- %r" % (np.mean(M_i),2*np.std(M_i)),M_i
    m,n = X.shape
    A = []
    B = []
    D = []
    R = []
    for j in range(n):
        X_j = X[:,j].ravel()
        res = stats.linregress(M_i, X_j)
        #print res
        (slope, intercept, r_value, p_value, 
            slope_std_error) = res
        B.append(intercept)
        A.append(slope)
        R.append(r_value)
        #D_j = X_j - intercept - M_i*slope
        D_j = X_j - M_i*slope
        D.append(np.mean(D_j))
    A = np.array(A).ravel()
    B = np.array(B).ravel()
    R = np.array(R).ravel()
    D = np.array(D).ravel()
    if verbose > 0:
        print "A:","mean(A):",np.mean(A),A.shape,A[:10]
        print "B:","mean(B):",np.mean(B),B.shape,B[:10]
        print "R:",R.shape,R[:10]
        print "D: mean(D):",np.mean(D),
        print "std(D):",np.std(D),"D:",D[:10]
    A = np.nan_to_num(A)
    B = np.nan_to_num(B)
    D = np.nan_to_num(D)
    return D,A,B

def ids_rmm(ids,X1):
    keys = sorted(ids)
    Xd = []; Xa = []; Xb = []
    for k in keys:
        v = ids[k]
        D,A,B = rmm(X1[v,:])
        Xd.append(D)
        Xa.append(A)
        Xb.append(B)
    return np.array(Xd),np.array(Xa),np.array(Xb)

def ids_rmm_parallel(ids,X1,k=4000):
    keys = sorted(ids)
    if k is not None:
        X2 = VarSel(k=k,std_ceil=0).fit_transform(X1)
    else:
        X2 = X1.copy()
    from joblib import Parallel, delayed
    pres = Parallel(n_jobs=-1)(delayed(rmm)(X2[ids[k],:]) for k in keys)
    Xd = []; Xa = []; Xb = []
    for (i,k) in enumerate(keys):
        D,A,B = pres[i]
        Xd.append(D)
        Xa.append(A)
        Xb.append(B)
    return np.array(Xd),np.array(Xa),np.array(Xb)

def test_rmm():
    m,n = 4,6
    X = np.zeros((m,n))
    for j in range(n):
        M_j = j+1
        for i in range(m):
            X[i,j] = (float(i)+1)/5 + (float(i+1)/5*M_j) + np.random.randn()/10
    rmm(X)
    ids = dict([('1',[1,3]),('2',[0,2])])
    Xd, Xa,Xb = ids_rmm(ids,X)
    print "Xd:",Xd
    Xd, Xa,Xb = ids_rmm_parallel(ids,X)
    print "Xd:",Xd

def test():
    test_rmm()
    print "tests ok"
 
if __name__ == '__main__':
    import random,sys
    random.seed(1)
    import argparse
    parser = argparse.ArgumentParser(description='Rmm.')
    parser.add_argument('cmd', nargs='?', default='test')
    args = parser.parse_args()
    print >>sys.stderr,args 
  
    if args.cmd == 'test':
        test()
    else:
        raise ValueError("bad cmd")

