# ***************************************************************************************************************************
# File      : Layers.py 
# Purpose   : Provides the defintion for Dense layer 
# Author    : Varun Gumma & Hanumantappa Budihal
#
# Date      : 09-02-2022 
# Bugs      : NA
# Change Log: 
#           -> 21-02-2022 : Added the function comments to all methods.
# ****************************************************************************************************************************/

import numpy as np
from .Activations import *

class Dense:
    def __init__(self, inp_dim, out_dim, activation="identity", initialization="xavier"):
        # parameters of the layer
        self.W = None 
        self.B = None 
        self.A = None 
        self.Z = None

        # gradients
        self.dW = None
        self.dB = None

        # a dictionary to map activation function name to corresponding function
        self.activation_f = {
            "identity" : identity,
            "sigmoid": sigmoid, 
            "tanh": tanh,
            "relu": ReLU,
            "leaky_relu": leaky_ReLU,
            "softmax": softmax
        }[activation]

        # initialize the parameters of the layer according to required initialization function
        if initialization == "random":
            self.W = np.random.random((inp_dim, out_dim))
            self.B = np.random.random((1, out_dim))
        elif initialization == "normal":
            self.W = np.random.normal(0, 1, size=(inp_dim, out_dim))
            self.B = np.random.normal(0, 1, size=(1, out_dim))
        elif initialization == "xavier":
            self.W = np.random.normal(0, np.sqrt(2/(inp_dim + out_dim)), size=(inp_dim, out_dim))
            self.B = np.random.normal(0, np.sqrt(2/(1 + out_dim)), size=(1, out_dim))
        elif initialization == "he":
            self.W = np.random.normal(0, np.sqrt(2/inp_dim), size=(inp_dim, out_dim))
            self.B = np.random.normal(0, np.sqrt(2), size=(1, out_dim))

    def __call__(self, x):
        """	    
        Forward propagates the input through the layer; y = f(wx + b)

        Parameters
	    ----------
        x :
            input to the layer
	    Returns
	    -------
            y -- activation of the layer, i.e. f(wx + b)
        """
        self.A = np.dot(x, self.W) + self.B 
        self.Z = self.activation_f(self.A)
        return self.Z

    def backward(self, dL, cache_z, weight_decay):
        """	    
        backprops the "error" through the layer.

        Parameters
	    ----------
        dL :
            error at the layer,
        cache_z :
             activation of the before layer
        weight_decay : L2 penalty value

	    Returns
	    -------
            "error" after backpropagating through the layer (sets the gradients in this process)
        """
        df = self.activation_f(self.A, derivative=True)
        # softmax derivative is a jacobian instead of a matrix
        # perform einstein-summation for it instead of element-wise multiplication
        self.dB = np.einsum('ijk,ik->ij', df, dL) if len(df.shape) > 2 else (df * dL)
        self.dW = np.dot(cache_z.T, self.dB) + (weight_decay * self.W)
        dL = np.dot(self.dB, self.W.T)
        self.dB = np.sum(self.dB, axis=0, keepdims=True)
        return dL

# ******************************************End of Dense class *************************************************************************#