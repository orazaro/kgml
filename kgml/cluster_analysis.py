#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author:   Oleg Razgulyaev 
# License:  BSD 3 clause
"""
    Cluster  analysis
"""
from __future__ import division
import csv, random, sys, datetime, os
import numpy as np
import pandas as pd
import warnings

#from config import datapath
datapath = 'data'
random_state = 1

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def do_pca(df, features, target, n_components=3, use_scaler=True):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    values = df[features].values
    if use_scaler:
        values = StandardScaler().fit_transform(values)
    pca = PCA(n_components=n_components, whiten=False)
    reduced_data = pca.fit_transform(values)
    pred_pca = ["PC{:d}".format(i+1) for i in range(reduced_data.shape[1])]

    df_pca = pd.DataFrame(reduced_data,columns=pred_pca)
    df_pca[target] = df[target].values

    return df_pca, pred_pca

def plot_pca(df, features, target, components=(0,1), figsize=(10,10)):
    """ Linear dimensionality reduction using Singular Value Decomposition 
        and plot the result.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    color = df[target]
    df_1 = df[features]
    #print df_1.columns
    values = df_1.values
    values = StandardScaler().fit_transform(values)
    reduced_data = PCA(n_components=max(components)+1,whiten=False).fit_transform(values)
    reduced_data = reduced_data[:,components]
    
    # use kgml lib
    from plot import rand_jitter
    rand_jitter2 = lambda x:rand_jitter(x,0.1)

    cmap = plt.get_cmap('bwr')
    df_2 = pd.DataFrame(reduced_data,columns=('PC1','PC2'))
    #print df.shape,np.unique(df['dplus'],return_counts=True)
    #df1 = df.apply(rand_jitter2, axis=0)
    df_3 = df_2
    xcolor = color

    fig,ax = plt.subplots(figsize=figsize)
    ax.scatter(df_3.iloc[:,0], df_3.iloc[:,1],c=xcolor,cmap=cmap,s=25)
    ax.set_xlabel('Principal Component {}'.format(components[0]+1))
    ax.set_ylabel('Principal Component {}'.format(components[1]+1))


def find_clusters(ax,reduced_data, n_clusters = 2, color='blue', cmap=plt.get_cmap('bwr'),
    title='K-means clustering on the dataset\n'
          'Centroids are marked with white cross'):
    """
    http://scikit-learn.sourceforge.net/dev/auto_examples/cluster/plot_kmeans_digits.html
    """

    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans.fit(reduced_data)

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
    y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()

    dx = (x_max - x_min) / 30
    dy = (y_max - y_min) / 30
    x_min, x_max = x_min - dx, x_max + dx
    y_min, y_max = y_min - dy, y_max + dy
    
    npixels = 500
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02    # point in the mesh [x_min, m_max]x[y_min, y_max].
    hx = (x_max - x_min) / npixels
    hy = (y_max - y_min) / npixels

    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    #plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], s=50, c=color, cmap=cmap)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    ax.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())

    return kmeans.predict(reduced_data)


# --------- Manifolds ---------------------------------#

def plot3d_manifold(df, target='appid', classes=['app0', 'app6', 'app23'],
                    xyz_labels=['T1', 'T2', 'T3'],
                    elev=None, azim=None, fsize=10, s=100, colors="ygr",
                    title="", ax2=None, legend_on=True):
    from mpl_toolkits.mplot3d import Axes3D

    if ax2 is None:
        fig = plt.figure(figsize=(fsize, fsize))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = ax2

    X_label = xyz_labels[0]
    Y_label = xyz_labels[1]
    Z_label = xyz_labels[2]

    for c, m, app in [
            (colors[0], 'o', classes[0]),
            (colors[1], 'o', classes[1]),
            (colors[2], 'o', classes[2])]:
        df1 = df[df[target] == app]
        xs = df1[X_label]
        ys = df1[Y_label]
        zs = df1[Z_label]
        ax.scatter3D(xs, ys, zs, c=c, marker=m, s=s, depthshade=False)

    ax.set_xlabel(X_label)
    ax.set_ylabel(Y_label)
    ax.set_zlabel(Z_label)
    ax.view_init(elev, azim)
    if title:
        ax.set_title(title)
    
    if legend_on:
        r_proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[0])
        g_proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[1])
        b_proxy = plt.Rectangle((0, 0), 1, 1, fc=colors[2])
        ax.legend([r_proxy, g_proxy, b_proxy],classes)
    
    if ax2 is None:
        plt.show()

def select_elev_azim(df, **args):
    fig  = plt.figure(figsize=(16, 64))

    i = 0
    for elev in range(0, 200, 20):
        for azim in range(0, 200, 20):
            i += 1
            ax = fig.add_subplot(20 ,5, i, projection='3d')
            args['elev'] = elev
            args['azim'] = azim
            args['title'] = "elev={} azim={}".format(elev, azim)
            args['ax2'] = ax
            args['legend_on'] = False
            args['s'] = 50
            plot3d_manifold(df, **args)



# --------- Tests -------------------------------------#

def test(args):
    print >>sys.stderr,"Test OK"

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Commands.')
    parser.add_argument('cmd', nargs='?', default='test',help="make_train|make_test")
    parser.add_argument('-rs', type=int, default=None,help="random_state")
    parser.add_argument('-fn', type=str, default='default.csv',help="filename to apply cmd")
    
    args = parser.parse_args()
    print >>sys.stderr,args 
    if args.rs:
        random_state = int(args.rs)
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)

    if args.cmd == 'test':
        test(args)
    elif args.cmd == 'make':
        pass
    else:
        raise ValueError("bad cmd")
    
