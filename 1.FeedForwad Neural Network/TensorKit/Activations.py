# ***************************************************************************************************************************
# File      : Activations.py 
# Purpose   : Provides access to the commonly used activation functions (sigmoid, ReLu,etc.,)
# Author    : Varun Gumma & Hanumantappa Budihal
#
# Date      : 08-02-2022 
# Bugs      : NA
# Change Log:
#            -> 21-02-2022 : Added the function comments to all methods.
# ****************************************************************************************************************************/

import numpy as np 

def identity(x, derivative=False):
    """
	The linear activation function or  Identity Function where the activation is proportional to the input

	Parameters
	----------
	x_value : 
		value for which function need to return the identity value
    derivative : boolean 
        Specifies the return value should derivative of function or not

	Returns
	-------        
    if derivative is false than return same 'x' value (f(x)=x)
    else a new array of given shape and data type, where the element’s value is set to 1
	"""
    if not derivative:
        return x 
    return np.ones(x.shape)

def sigmoid(x_value, derivative=False):
    """
	This is especially used for model where we have to predict the probability as an out.
    Since probability of anything exists only between the range of 0 and 1.

	Parameters
	----------
	x_value : 
		value for which function need to return the sigmoid value
    derivative : boolean 
        Specifies the return value should derivative of function or not

	Returns
	-------
    If derivative is false: 
        then function returns y=(1/1+e^(-x)) for  ||x|| <= 10 else returns 1 (this is because if x_values cross 10 
        y values is almost equal 1, added this condition to avoid the overflow - numerical safe function) 
    Else y * (1-y) derivate value of sigmoid
	"""
    vectorized_fucntion = np.vectorize(lambda x : 1. if x > 10 else 0.)
    y_value = np.where(np.abs(x_value) > 10, vectorized_fucntion(x_value), 1/(1 + np.exp(-x_value)))

    if not derivative:
        return y_value
    return y_value * (1 - y_value)

def tanh(x_value, derivative=False):
    """
	tanh is also like logistic sigmoid but better.The range of the tanh function is from (-1 to 1).
    tanh is also sigmoidal (s - shaped).
    The negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.

	Parameters
	----------
	x_value 
		value for which function need to return the tanh value
    derivative : boolean 
        Specifies the return value should derivative of function or not

	Returns
	-------       
    If derivative is false: 
        then function returns tanh value       
    Else derivate value of tanh
    """
    y_value = np.tanh(x_value)

    if not derivative:
        return y_value
    return 1 - np.square(y_value)

def ReLU(x_value, derivative=False):
    """
	ReLU is half rectified (from bottom). 
    f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

	Parameters
	----------
	x_value 
		value for which function need to return the ReLU value
    derivative : boolean 
        Specifies the return value should derivative of function or not

	Returns
	-------       
    If derivative is false: 
        then function returns max{0, x_value}     
    Else 1 if x_value is greater than 0 else 0
    """
    if not derivative:
        return np.where(x_value > 0, x_value, 0)
    return np.where(x_value > 0, 1, 0)

def leaky_ReLU(x_value, alpha=0.01, derivative=False):
    """
	Leaky ReLU, is a type of activation function based on a ReLU,
    but it has a small slope for negative values instead of a flat slope.

	Parameters
	----------
	x_value :
		value for which function need to return the leaky Relu value
    alpha : slope coefficient
    derivative : boolean 
        Specifies the return value should derivative of function or not.

    Returns
	-------       
    If derivative is false: 
        then function returns x_value if x_values > 0 else alpha*x_value      
    Else return 1 if x_values > 0 else alpha
    """
    if not derivative:
        return np.where(x_value > 0, x_value, alpha*x_value)
    return np.where(x_value > 0, 1, alpha)

def softmax(x, derivative=False):
    """
	The softmax activation function calculates the relative probalities

	Parameters
	----------
	x_value 
		value for which function need to return the softmax value
    derivative : boolean 
        Specifies the return value should derivative of function or not

	Returns
	-------       
    If derivative is false: 
        then function returns exponential divided by the sum of exponential of the whole x value:     
    Else return derivates value of the softmax
    """
    exponentials = np.exp(x - x.max(axis=1, keepdims=True))
    softmax_value = exponentials/exponentials.sum(axis=1, keepdims=True)
    if not derivative:
        return softmax_value
    return np.einsum('ij,jk->ijk', softmax_value, np.eye(x.shape[1])) - \
           np.einsum('ij,ik->ijk', softmax_value, softmax_value)

# ******************************************End of Activations.py*************************************************************************#