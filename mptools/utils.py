################################################################################
## Basic utility functions  #########################
################################################################################

import numpy as np
from numpy import asarray as arr
from numpy import atleast_2d as twod
import matplotlib.pyplot as plt
import pickle

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
    


def bootstrapData(X, Y=None, n_boot=None):
    """
    Bootstrap resample a data set (with replacement): 
    draw data points (x_i,y_i) from (X,Y) n_boot times.

    Parameters
    ----------
    X : MxN numpy array of data points to be resampled.
    Y : Mx1 numpy array of labels associated with each datum (optional)
    n_boot : int, number of samples to draw (default: M)

    Returns
    -------
    Xboot, Yboot : (tuple of) numpy arrays for the resampled data set
    If Y is not present or None, returns only Xboot (non-tuple)
    """
    nx,dx = twod(X).shape
    if n_boot is None: n_boot = nx
    idx = np.floor(np.random.rand(n_boot) * nx).astype(int)
    if Y is None: return X[idx,:]
    Y = Y.flatten()
    assert nx == len(Y), 'bootstrapData: X and Y should have the same length'
    return (X[idx,:],Y[idx])



def combTheta(pklName,startIdx=0,endIdx=0,saveTheta=False,saveName='theta'):
    """
    Combine multiple inferred theta (vertically) generated from parallel computing.
    datapoints contain nan will be dropped.
    
    Parameters
    ----------
    pklName : repository, file location + file name, in string type.
    startIdx : starting index label of the pickle variable, included (default: 0).
    endIdx : ending index label of the pickle variable, included (default: 0).
    saveTheta : save this combined theta as a new pickle file (optional).
    saveName : if save, the name of the pickle file (optional, default: theta).
    
    Returns
    -------
    Theta : combined theta, in numpy array type.
    """
    pickle_in = open(pklName+str(startIdx)+'.pkl',"rb")
    Theta = pickle.load(pickle_in)
    Theta = Theta[~np.isnan(Theta).any(axis=1)]
    for i in range(startIdx+1,endIdx+1):
        pickle_in = open(pklName+str(i)+'.pkl',"rb")
        Theta_temp = pickle.load(pickle_in)
        Theta = np.vstack((Theta,Theta_temp[~np.isnan(Theta_temp).any(axis=1)]))
    if saveTheta:
        pickle_out = open(saveName+'.pkl',"wb")
        pickle.dump(Theta, pickle_out)
        pickle_out.close()
    return Theta



def empiricalRule(Theta,pct=.68):
    """
    Applied empirical rule on the data for analysis.
    
    Parameters
    ----------
    Theta : MxN numpy array of combined theta values.
    pct : percentage in the middle that will be preserved, range from 0 to 1.
    
    Returns
    -------
    Theta : truncated and sorted theta.
    mean, standard deviation, minimal value, maximal value.
    """
    n = Theta.shape[0]
    tileLow = (1-pct)/2
    tileHigh = 1-tileLow
    Theta.sort(axis=0)
    Theta = Theta[round(tileLow*n):round(tileHigh*n)]
    return Theta, np.mean(Theta,axis=0), np.std(Theta,axis=0), Theta[0], Theta[-1]

################################################################################
################################################################################
################################################################################
