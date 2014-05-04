#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Imbalanced datasets    
"""
import numpy as np
import random

def round_down(Xall,y1):
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = random.sample(p_zeros,len(p_ones))
        sel = sel + p_ones
    elif delta < 0:
        sel = random.sample(p_ones,len(p_zeros))
        sel = sel + p_zeros
    else:
        return Xall,y1
    print "round down:",len(p_zeros),len(p_ones),len(sel)
    return Xall[sel,:],y1[sel]

def round_up(Xall,y1,ids):
    ids_inv = [None]*Xall.shape[0]
    for (k,v) in ids.iteritems():
        for i in v:
            ids_inv[i] = k
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        sel = [random.choice(p_ones) for _ in range(delta)]
    elif delta < 0:
        delta = -delta
        sel = [random.choice(p_zeros) for _ in range(delta)]
    else:
        return Xall,y1,ids
    X1 = [Xall]
    z1 = list(y1)
    j = Xall.shape[0]
    for i in sel:
        X1.append(Xall[i,:])
        z1.append(y1[i])
        ids[ids_inv[i]].append(j)
        j += 1
    X1 = np.vstack(X1)
    z1 = np.array(z1).ravel()
    print "round_up: 0:",len(p_zeros),"1:",len(p_ones),"X1:",X1.shape,"y1:",z1.shape,"j_last:",j
    return X1,z1,ids

from smote import SMOTE

def round_smote(Xall,y1,k=5,h=1.0):
    p_zeros = [i for i,e in enumerate(y1) if e == 0]
    p_ones = [i for i,e in enumerate(y1) if e > 0]
    delta = len(p_zeros) - len(p_ones)
    if delta > 0:
        N = ( int(len(p_zeros)/len(p_ones))+1 ) * 100
        T = Xall[p_ones,:]
        S = SMOTE(T, N, k, h):
        sel = random.sample(range(S.shape[0]),delta)
        X1 = np.vstack([Xall,S[sel,:]])
        z1 = np.hstack(y1,np.ones(delta))
    elif delta < 0:
        delta = -delta
        N = ( int(len(p_ones)/len(p_zeros))+1 ) * 100
        T = Xall[p_zeros,:]
        S = SMOTE(T, N, k, h):
        sel = random.sample(range(S.shape[0]),delta)
        X1 = np.vstack([Xall,S[sel,:]])
        z1 = np.hstack(y1,np.zeros(delta))
    else:
        return Xall,y1
    print "round smote:","X1:",X1.shape,"z1:",z1.shape
    return X1,z1

