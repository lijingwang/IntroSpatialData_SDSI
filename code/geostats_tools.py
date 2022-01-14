#!/usr/bin/env python
# coding: utf-8


### geostatistical tools
# Mickey MacKie

# Modified by Lijing Wang (2021/01/13)
## Address the negative weight problem
## Note: In kriging, it is ok to have negative weights, if the available data locations are too close
## However, that usually ends up giving unrealistic values
## To mitigate this problem, we ignore data points with negative weights and recalculate the kriging weights


import numpy as np
import numpy.linalg as linalg
import pandas as pd
import sklearn as sklearn
from sklearn.neighbors import KDTree
import math
from scipy.spatial import distance_matrix
from tqdm import tqdm
import random


# covariance function definition
def covar(t, d, r):
    h = d / r
    if t == 1:  # Spherical
        c = 1 - h * (1.5 - 0.5 * np.square(h))
        c[h > 1] = 0
    elif t == 2:  # Exponential
        c = np.exp(-3 * h)
    elif t == 3:  # Gaussian
        c = np.exp(-3 * np.square(h))
    return c


# get variogram along the major or minor axis
def axis_var(lagh, nug, nstruct, cc, vtype, a):
    lagh = lagh
    nstruct = nstruct # number of variogram structures
    vtype = vtype # variogram types (Gaussian, etc.)
    a = a # range for axis in question
    cc = cc # contribution of each structure
    
    n = len(lagh)
    gamma_model = np.zeros(shape = (n))
    
    # for each lag distance
    for j in range(0,n):
        c = nug
        h = np.matrix(lagh[j])
        
        # for each structure in the variogram
        for i in range(nstruct):
            Q = h.copy()
            d = Q / a[i]
            c = c + covar(vtype[i], d, 1) * cc[i] # covariance
        
        gamma_model[j] = 1+ nug - c # variance
    return gamma_model



# make array of x,y coordinates based on corners and resolution
def pred_grid(xmin, xmax, ymin, ymax, pix):
    cols = (xmax - xmin)/pix; rows = (ymax - ymin)/pix  # number of rows and columns
    x = np.arange(xmin,xmax,pix); y = np.arange(ymin,ymax,pix) # make arrays

    xx, yy = np.meshgrid(x,y) # make grid
    yy = np.flip(yy) # flip upside down

    # shape into array
    x = np.reshape(xx, (int(rows)*int(cols), 1))
    y = np.reshape(yy, (int(rows)*int(cols), 1))

    Pred_grid_xy = np.concatenate((x,y), axis = 1) # combine coordinates
    return Pred_grid_xy # returns numpy array of x,y coordinates



# rotation matrix (Azimuth = major axis direction)
def Rot_Mat(Azimuth, a_max, a_min):
    theta = (Azimuth / 180.0) * np.pi
    Rot_Mat = np.dot(
        np.array([[1 / a_max, 0], [0, 1 / a_min]]),
        np.array(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        ),
    )
    return Rot_Mat


# covariance model
def cov(h1, h2, k, vario):
    # unpack variogram parameters
    Azimuth = vario[0]
    nug = vario[1]
    nstruct = vario[2]
    vtype = vario[3]
    cc = vario[4]
    a_max = vario[5]
    a_min = vario[6]
    
    c = nug 
    for i in range(nstruct):
        Q1 = h1.copy()
        Q2 = h2.copy()
        
        # covariances between measurements
        if k == 0:
            d = distance_matrix(
                np.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])),
                np.matmul(Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])),
            )
            
        # covariances between measurements and unknown
        elif k == 1:
            d = np.sqrt(
                np.square(
                    (np.matmul(Q1, Rot_Mat(Azimuth, a_max[i], a_min[i])))
                    - np.tile(
                        (
                            np.matmul(
                                Q2, Rot_Mat(Azimuth, a_max[i], a_min[i])
                            )
                        ),
                        (k, 1),
                    )
                ).sum(axis=1)
            )
            d = np.asarray(d).reshape(len(d))
        sill = np.sum(cc)+nug
        gamma =  sill - covar(vtype[i], d, 1) * cc[i]
        c =  sill-gamma
    return c

# sequential Gaussian simulation
def sgsim(Pred_grid, df, xx, yy, data, k, vario):

    """Sequential Gaussian simulation
    :param Pred_grid: x,y coordinate numpy array of prediction grid
    :df: data frame of input data
    :xx: column name for x coordinates of input data frame
    :yy: column name for y coordinates of input data frame
    :df: column name for the data variable (in this case, bed elevation) of the input data frame
    :k: the number of conditioning points to search for
    :vario: variogram parameters describing the spatial statistics
    """
    
    # generate random array for simulation order
    xyindex = np.arange(len(Pred_grid))
    random.shuffle(xyindex)

    Var_1 = np.var(df[data]); # variance of data 
    
    # preallocate space for simulation
    sgs = np.zeros(shape=len(Pred_grid))
    
    # preallocate space for data
    #X_Y = np.zeros((1, k, 2))
    #closematrix_Primary = np.zeros((1, k))
    
    with tqdm(total=len(Pred_grid), position=0, leave=True) as pbar:
        for i in tqdm(range(0, len(Pred_grid)), position=0, leave=True):
            X_Y = np.zeros((1, k, 2))
            closematrix_Primary = np.zeros((1, k))
    
            pbar.update()
            z = xyindex[i]
            
            # make KDTree to search data for nearest neighbors
            tree_data = KDTree(df[[xx,yy]].values) 
    
            # find nearest data points
            nearest_dist, nearest_ind = tree_data.query(Pred_grid[z : z + 1, :], k=k)
            a = nearest_ind.ravel()
            group = df.iloc[a, :]
            closematrix_Primary[:] = group[data]
            X_Y[:, :] = group[[xx, yy]]
        
            # left hand side (covariance between data)
            Kriging_Matrix = np.zeros(shape=((k+1, k+1)))
            Kriging_Matrix[0:k,0:k] = cov(X_Y[0], X_Y[0], 0, vario)
            Kriging_Matrix[k,0:k] = 1
            Kriging_Matrix[0:k,k] = 1
        
            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(k+1))
            k_weights = r
            r[0:k] = cov(X_Y[0], np.tile(Pred_grid[z], (k, 1)), 1, vario)
            r[k] = 1 # unbiasedness constraint
            Kriging_Matrix.reshape(((k+1)), ((k+1)))
        
            # Calculate Kriging Weights
            k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

            # get estimates
            est = np.sum(k_weights[0:k]*closematrix_Primary[:]) # kriging mean
            var = Var_1 - np.sum(k_weights[0:k]*r[0:k]) # kriging variance
        

            while np.sum(k_weights[:-1]<0):
                new_k = np.sum(k_weights[:-1]>=0)
                new_k_idx = np.where(k_weights[:-1]>=0)[0]
                Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
                Kriging_Matrix[0:new_k,0:new_k] = cov(X_Y[0][new_k_idx], X_Y[0][new_k_idx], 0, vario)
                Kriging_Matrix[new_k,0:new_k] = 1
                Kriging_Matrix[0:new_k,new_k] = 1

                # Set up Right Hand Side (covariance between data and unknown)
                r = np.zeros(shape=(new_k+1))
                k_weights = r
                r[0:new_k] = cov(X_Y[0][new_k_idx], np.tile(Pred_grid[z], (new_k, 1)), 1, vario)
                r[new_k] = 1 # unbiasedness constraint
                Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))
                
                # Calculate Kriging Weights
                k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

                # get estimates
                est = np.sum(k_weights[0:new_k]*closematrix_Primary[:][:,new_k_idx]) # kriging mean
                var = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k]) # kriging variance

                closematrix_Primary = closematrix_Primary[:,new_k_idx]
                X_Y = X_Y[:,new_k_idx]


            if (var < 0): # make sure variances are non-negative
                var = 0

            sgs[z] = np.random.normal(est,math.sqrt(var),1) # simulate by randomly sampling a value
        
            # update the conditioning data
            coords = Pred_grid[z:z+1,:]
            dnew = {xx: [coords[0,0]], yy: [coords[0,1]], data: [sgs[z]]} 
            dfnew = pd.DataFrame(data = dnew)
            df = pd.concat([df,dfnew], sort=False) # add new points by concatenating dataframes 
        
    return sgs



# simple kriging
def skrige(Pred_grid, df, xx, yy, data, k, vario):
    
    Mean_1 = np.average(df[data]) # mean of data
    Var_1 = np.var(df[data]); # variance of data 
    
    # make KDTree to search data for nearest neighbors
    tree_data = KDTree(df[[xx,yy]].values) 
    
    # preallocate space for mean and variance
    est_SK = np.zeros(shape=len(Pred_grid))
    var_SK = np.zeros(shape=len(Pred_grid))
    
    # preallocate space for data
    #X_Y = np.zeros((1, k, 2))
    #closematrix_Primary = np.zeros((1, k)) 
    neardistmatrix = np.zeros((1, k))
    
    for z in tqdm(range(0, len(Pred_grid))):
        
        X_Y = np.zeros((1, k, 2))
        closematrix_Primary = np.zeros((1, k)) 
        # find nearest data points
        nearest_dist, nearest_ind = tree_data.query(Pred_grid[z : z + 1, :], k=k)
        a = nearest_ind.ravel()
        group = df.iloc[a, :]
        closematrix_Primary[:] = group[data]
        neardistmatrix[:] = nearest_dist
        X_Y[:, :] = group[[xx, yy]]
        
        # left hand side (covariance between data)
        Kriging_Matrix = np.zeros(shape=((k, k)))
        Kriging_Matrix = cov(X_Y[0], X_Y[0], 0, vario)
        
        # Set up Right Hand Side (covariance between data and unknown)
        r = np.zeros(shape=(k))
        k_weights = r
        r = cov(X_Y[0], np.tile(Pred_grid[z], (k, 1)), 1, vario)
        Kriging_Matrix.reshape(((k)), ((k)))
        
        # Calculate Kriging Weights
        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est_SK[z] = k*Mean_1 + np.sum(k_weights*(closematrix_Primary[:] - Mean_1))
        var_SK[z] = Var_1 - np.sum(k_weights*r)

        # Address the negative weight problem:
        ## If the value of the weight is less than 0, we ignore this data and only use the rest of them
        while np.sum(k_weights<0):
            new_k = np.sum(k_weights>=0)
            new_k_idx = np.where(k_weights>=0)[0]
            Kriging_Matrix = np.zeros(shape=((new_k, new_k)))
            Kriging_Matrix = cov(X_Y[0][new_k_idx], X_Y[0][new_k_idx], 0, vario)
            
            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k))
            k_weights = r
            r = cov(X_Y[0][new_k_idx], np.tile(Pred_grid[z], (new_k, 1)), 1, vario)
            Kriging_Matrix.reshape(((new_k)), ((new_k)))
            
            # Calculate Kriging Weights
            k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

            # get estimates
            est_SK[z] = k*Mean_1 + np.sum(k_weights*(closematrix_Primary[:][:,new_k_idx] - Mean_1))
            var_SK[z] = Var_1 - np.sum(k_weights*r)

            closematrix_Primary = closematrix_Primary[:,new_k_idx]
            X_Y = X_Y[:,new_k_idx]
            

    return est_SK, var_SK


# ordinary kriging
def okrige(Pred_grid, df, xx, yy, data, k, vario):
    
    Var_1 = np.var(df[data]); # variance of data 
    
    # make KDTree to search data for nearest neighbors
    tree_data = KDTree(df[[xx,yy]].values) 
    
    # preallocate space for mean and variance
    est_OK = np.zeros(shape=len(Pred_grid))
    var_OK = np.zeros(shape=len(Pred_grid))
    
    # preallocate space for data
    # X_Y = np.zeros((1, k, 2))
    # closematrix_Primary = np.zeros((1, k))
    neardistmatrix = np.zeros((1, k))
    
    for z in tqdm(range(0, len(Pred_grid))):
        X_Y = np.zeros((1, k, 2))
        closematrix_Primary = np.zeros((1, k))

        # find nearest data points
        nearest_dist, nearest_ind = tree_data.query(Pred_grid[z : z + 1, :], k=k)
        a = nearest_ind.ravel()
        group = df.iloc[a, :]
        closematrix_Primary[:] = group[data]
        neardistmatrix[:] = nearest_dist
        X_Y[:, :] = group[[xx, yy]]
        
        # left hand side (covariance between data)
        Kriging_Matrix = np.zeros(shape=((k+1, k+1)))
        Kriging_Matrix[0:k,0:k] = cov(X_Y[0], X_Y[0], 0, vario)
        Kriging_Matrix[k,0:k] = 1
        Kriging_Matrix[0:k,k] = 1
        
        # Set up Right Hand Side (covariance between data and unknown)
        r = np.zeros(shape=(k+1))
        k_weights = r
        r[0:k] = cov(X_Y[0], np.tile(Pred_grid[z], (k, 1)), 1, vario)
        r[k] = 1 # unbiasedness constraint
        Kriging_Matrix.reshape(((k+1)), ((k+1)))
        
        # Calculate Kriging Weights
        k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

        # get estimates
        est_OK[z] = np.sum(k_weights[0:k]*closematrix_Primary[:])
        var_OK[z] = Var_1 - np.sum(k_weights[0:k]*r[0:k])


        # Address the negative weight problem:
        ## If the value of the weight is less than 0, we ignore this data and only use the rest of them
        while np.sum(k_weights[:-1]<0):
            new_k = np.sum(k_weights[:-1]>=0)
            new_k_idx = np.where(k_weights[:-1]>=0)[0]
            Kriging_Matrix = np.zeros(shape=((new_k+1, new_k+1)))
            Kriging_Matrix[0:new_k,0:new_k] = cov(X_Y[0][new_k_idx], X_Y[0][new_k_idx], 0, vario)
            Kriging_Matrix[new_k,0:new_k] = 1
            Kriging_Matrix[0:new_k,new_k] = 1

            # Set up Right Hand Side (covariance between data and unknown)
            r = np.zeros(shape=(new_k+1))
            k_weights = r
            r[0:new_k] = cov(X_Y[0][new_k_idx], np.tile(Pred_grid[z], (new_k, 1)), 1, vario)
            r[new_k] = 1 # unbiasedness constraint
            Kriging_Matrix.reshape(((new_k+1)), ((new_k+1)))
            
            # Calculate Kriging Weights
            k_weights = np.dot(np.linalg.pinv(Kriging_Matrix), r)

            # get estimates
            est_OK[z] = np.sum(k_weights[0:new_k]*closematrix_Primary[:][:,new_k_idx]) # kriging mean
            var_OK[z] = Var_1 - np.sum(k_weights[0:new_k]*r[0:new_k]) # kriging variance

            closematrix_Primary = closematrix_Primary[:,new_k_idx]
            X_Y = X_Y[:,new_k_idx]
        
    return est_OK, var_OK






