## IMPORTS #####################################################################
import math
import numpy as np

from numpy import asarray as arr

from .utils import toIndex

import torch
import matplotlib.pyplot as plt

from IPython import display

################################################################################
## Base (abstract) "classify" class and associated functions ###################
################################################################################

class jointClassifier:

  def __init__(self, *args, **kwargs):
    """Constructor for abstract base class for various classifiers. 

    This class implements methods that generalize to different classifiers.
    Optional arguments X,Y,... call train(X,Y,...) to initialize the model
    """
    self.classes = []
    if len(args) or len(kwargs):
        return self.train(*args, **kwargs)


  def __call__(self, *args, **kwargs):
    """Provides syntatic sugar for prediction; calls "predict".  """ 
    return self.predict(*args, **kwargs)


  def predict(self,X):
    '''Predict model on data X; return nparray of class predictions'''
    N1 = 104 # length of CRP binding sites
    
    X1 = X[:,:N1] # X1 is a CRP sequence
    X2 = X[:,N1:] # X2 is a RNAP sequence
    
    R = 1.98e-3 # gas const
    T = 310 # temperature at which cells were induced
    RT = R*T

    eps_c = torch.Tensor(X1) @ self.theta[4:4+N1]; # CRP binding energy
    eps_r = torch.Tensor(X2) @ self.theta[4+N1:]; # RNAP binding energy
    wi = self.theta[2]*(torch.exp(-eps_r/RT)+self.theta[1]*torch.exp(-(eps_c+eps_r+self.theta[3])/RT))
    ri = wi/(1+self.theta[1]*torch.exp(-eps_c/RT)+wi)-self.theta[0]

    Y01 = 1*(ri>0)                   # binary classification threshold; convert to integers
    Y = self.classes[Y01]           # use lookup to convert back to class values if given
    return Y                        # NOTE: returns as numpy, not torch! (b/c classes is a nparray)
                                    # (This is necessary for mltools plot to work)

  def train(self,X,Y,initStep=1.,stopTol=1e-4,stopEpochs=5000,alpha=0):
    """ Train the logistic regression using stochastic gradient descent """
    
    M,N = X.shape;                     # initialize the model if necessary:
    N1 = 104 # length of CRP binding sites
    # N2 = N-N1;
    X1 = X[:,:N1] # X1 is a CRP sequence
    X2 = X[:,N1:] # X2 is a RNAP sequence
    self.classes = np.unique(Y);       # Y may have two classes, any values
    # Y01 = ml.toIndex(Y,self.classes);  # Y01 is Y, but with canonical values 0 or 1
    Y01 = Y
    
    R = 1.98e-3 # gas const
    T = 310 # temperature at which cells were induced
    RT = R*T
    
    N_theta = 4+N # the number of unknown parameters
    # if the shape of initial theta is wrong, randomly generate a correct one
    if len(self.theta)!=N_theta: self.theta=torch.randn((N_theta,1),requires_grad = True);
    
        
    # init loop variables:
    epoch=0; done=False; Jnll=[]; J01=[];            # initialize loop variables
    myrate = lambda epoch: initStep*2.0/(2.0+epoch)  # step size as a f'n of epoch
    
    opt = torch.optim.SGD([self.theta], initStep)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, myrate)
    
    
    while not done:
        # Do an SGD pass through the entire data set:
        Jnll.append(0.)
        for i in np.random.permutation(M):
            # Compute predictions and loss for *just* data X[i]:
            # Note: @ operation requires Python 3.5+
            eps_c = torch.tensor(X1[i],dtype=torch.float32) @ self.theta[4:4+N1]; # CRP binding energy
            eps_r = torch.tensor(X2[i],dtype=torch.float32) @ self.theta[4+N1:]; # RNAP binding energy
            wi = self.theta[2]*(torch.exp(-eps_r/RT)+self.theta[1]*torch.exp(-(eps_c+eps_r+self.theta[3])/RT))
            ri = wi/(1+self.theta[1]*torch.exp(-eps_c/RT)+wi)-self.theta[0]
            si = 1/(1+torch.exp(-ri)); # logistic (probability) prediction of the class
            Ji_pre = -Y01[i]*torch.log(si)-(1-Y01[i])*torch.log(1-si); # torch.Tensor shape [] (scalar)
            # Ji_pre = torch.maximum( 1-(2*Y01[i]-1)*ri, 0*ri); # Hinge loss
            
            # add penalty to constrain the range of parameters
            Ji0 = Ji_pre + alpha*torch.maximum( self.theta[0]*(self.theta[0]-1), 0*self.theta[0]); 
            Ji1 = Ji0 + alpha*torch.maximum( self.theta[1]*(self.theta[1]-1), 0*self.theta[1]);
            Ji = Ji1 + alpha*torch.maximum( self.theta[2]*(self.theta[2]-1), 0*self.theta[2]);
            
            Jnll[-1] += float(Ji)/M             # find running average of surrogate loss
            opt.zero_grad()                     # Ji should be a torch.tensor of shape []
            Ji.backward()
            opt.step()
        sched.step()        

        epoch += 1

        J01.append( self.err(X,Y) )  # evaluate the current actual error rate 

        display.clear_output(wait=True); plt.figure(figsize=(15,5));
        plt.subplot(1,2,1); 
        plt.cla(); plt.plot(Jnll,'b-'); plt.xlabel('epoch'); plt.ylabel('surrogate loss');    # plot losses
        plt.subplot(1,2,2); 
        plt.cla(); plt.plot(J01,'r-'); plt.xlabel('epoch'); plt.ylabel('error rate');    # plot error rate
        plt.show(); plt.pause(.001);                    # let OS draw the plot

        ## For debugging: you may want to print current parameters & losses
        # print(self.theta, ' => ', Jnll, ' / ', J01[-1] )
        # input()   # pause for keystroke

        # check stopping criteria: exit if exceeded # of epochs ( > stopEpochs)
        done = epoch > stopEpochs or abs(Jnll[-1]) < stopTol;   # or if Jnll not changing between epochs ( < stopTol )



  ####################################################
  # Standard loss f'n definitions for classifiers    #
  ####################################################
  def err(self, X, Y):
    """This method computes the error rate on a data set (X,Y)

    Args: 
        X (arr): M,N array of M data points with N features each
        Y (arr): M, or M,1 array of target class values for each data point

    Returns:
        float: fraction of prediction errors, 1/M \sum (Y[i]!=f(X[i]))
    """
    Y    = arr( Y )
    Yhat = arr( self.predict(X) )
    return np.mean(Yhat.reshape(Y.shape) != Y)


################################################################################
################################################################################
################################################################################
