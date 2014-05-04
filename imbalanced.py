#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Imbalanced datasets    
"""
import numpy as np
import random

def round_down(Xall,y1,y2):
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
        return Xall,y1,y2
    print "round down:",len(p_zeros),len(p_ones),len(sel)
    z2 = np.array([random.randint(0,1) for _ in range(len(sel))]).ravel()
    return Xall[sel,:],y1[sel],z2

def round_up(Xall,y1,y2,ids):
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
        return Xall,y1,y2,ids
    X1 = [Xall]
    z1 = list(y1)
    z2 = list(y2)
    j = Xall.shape[0]
    for i in sel:
        X1.append(Xall[i,:])
        z1.append(y1[i])
        z2.append(y2[i])
        ids[ids_inv[i]].append(j)
        j += 1
    X1 = np.vstack(X1)
    z1 = np.array(z1).ravel()
    #z2 = np.array(z2).ravel()
    z2 = np.array([random.randint(0,1) for _ in range(len(z2))]).ravel()
    print "round_up: 0:",len(p_zeros),"1:",len(p_ones),"X1:",X1.shape,"y1:",z1.shape,"y2:",z2.shape,"j_last:",j

    return X1,z1,z2,ids
