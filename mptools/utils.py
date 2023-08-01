################################################################################
## Basic utility functions  #########################
################################################################################

import numpy as np
from numpy import asarray as arr
from numpy import atleast_2d as twod
import matplotlib.pyplot as plt


def bp2bi(sq):
    """
    base pairs to binary data
    """
    l = []
    for bp in sq:
        if bp == 'A':
            l += [1,0,0,0]
        elif bp == 'C':
            l += [0,1,0,0]
        elif bp == 'G':
            l += [0,0,1,0]
        elif bp == 'T':
            l += [0,0,0,1]
        else:
            print('Error: please ensure only ACGT nucliotides contained.')
            return
    return l



def seq2data(sqs,bin_min,bin_max,pt_min,pt_max):
    """
    sequences to data in the standard form: feature values + label value
    """
    data = []
    for k in range(bin_min,bin_max+1):
        for sq in sqs[k]:
            l = bp2bi(sq[pt_min:pt_max+1]) if pt_max != -1 else bp2bi(sq[pt_min:])
            l.append(k)
            data.append(l)
    return np.array(data,np.float32)



def shuffleData(X, Y=None):
    """
    Shuffle (randomly reorder) data in X and Y.

    Parameters
    ----------
    X : MxN numpy array: N feature values for each of M data points
    Y : Mx1 numpy array (optional): target values associated with each data point

    Returns
    -------
    X,Y  :  (tuple of) numpy arrays of shuffled features and targets
            only returns X (not a tuple) if Y is not present or None
    
    Ex:
    X2    = shuffleData(X)   : shuffles the rows of the data matrix X
    X2,Y2 = shuffleData(X,Y) : shuffles rows of X,Y, preserving correspondence
    """
    nx,dx = twod(X).shape
    Y = arr(Y).flatten()
    ny = len(Y)

    pi = np.random.permutation(nx)
    X = X[pi,:]

    if ny > 0:
        assert ny == nx, 'shuffleData: X and Y must have the same length'
        Y = Y[pi] if Y.ndim <= 1 else Y[pi,:]
        return X,Y

    return X



def toIndex(Y, values=None):
    """
    Function that converts discrete value Y into [0 .. K - 1] (index) 
    representation; i.e.: toIndex([4 4 1 1 2 2], [1 2 4]) = [2 2 0 0 1 1].

    Parameters
    ----------
    Y      : (M,) or (M,1) array-like of values to be converted
    values : optional list that specifices the value/index mapping to use for conversion.

    Returns
    -------
    idx    : (M,) or (M,1) array that contains indexes instead of values.
    """
    n,d = np.matrix(Y).shape

    assert min(n,d) == 1
    values = list(values) if values is not None else list(np.unique(Y))
    C = len(values)
    #flat_Y = Y.ravel()

    idx = []
    for v in Y:
        idx.append(values.index(v))
    return np.array(idx)



def energyMatrix(theta_raw, plot = False):
    """
    generate energy matrix, the normalized energy vector and related constants
    input theta_raw has NO intercept term (pure energy vector with no shifting term)
    """
    n = int((theta_raw.shape[0])/4)
    theta_shift = 0 # initialize energy shift
    theta = np.transpose(np.reshape(theta_raw,(n,4)))
    for i in range(n):
        theta_shift += min(theta[:,i]) # cummulate energy shift
        theta[:,i] -= min(theta[:,i])       
    theta_scale = theta.max() # calculate the unit/scaling factor of energy
    theta = theta/theta_scale
    theta_long = np.reshape(np.transpose(theta),theta_raw.shape)
    if plot:
        plt.matshow(theta)
        plt.colorbar(label='arbitrary unit')
        plt.style.use('classic')
        plt.show()
    return theta, theta_long, theta_shift, theta_scale

################################################################################
################################################################################
################################################################################
