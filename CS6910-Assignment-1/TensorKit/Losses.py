
# ***************************************************************************************************************************
# File      : Losses.py 
# Purpose   : Provides access to the commonly used loss functions  
# Author    : Varun Gumma & Hanumantappa Budihal
#
# Date      : 08-02-2022 
# Bugs      : NA
# Change Log: 21-02-2022 : Added the function comments to all methods.
# ****************************************************************************************************************************/

import numpy as np

def MeanSquaredLoss(y, t, derivative=False):
    """  
    Finds the mse and its derivative between the target and prediction.

    Parameters
	----------
	y : prediction
    t : target
    derivative : if derivative is to be calculated
    Returns
	-------
        mse scalar value or the derivative vector (as per the bool variable derivative)     
    """
    if not derivative:
        return 0.5 * np.sum(np.square(y - t))
    return y - t

def BinaryCrossEntropy(y, t, derivative=False):
    """  
    Finds the bce and its derivative between the target and prediction

    Parameters
	----------
	y : prediction
    t : target
    derivative : if derivative is to be calculated

    Returns
	-------
        bce scalar value or the derivative vector (as per the bool variable derivative)      
    """
    if not derivative:
        return -np.sum((t * np.log(y + 1e-8)) + ((1 - t) * np.log(1 - y + 1e-8)))
    return -(t/(y + 1e-8) - (1 - t)/(1 - y + 1e-8))

def CategoricalCrossEntropy(y, t, derivative=False):
    """  
    Finds the cce and its derivative between the target and prediction.

    Parameters
	----------
	y : prediction
    t : target
    derivative : if derivative is to be calculated

    Returns
	-------
        cce scalar value or the derivative vector (as per the bool variable derivative)    
    """
    if not derivative:
        return -np.sum(t * np.log(y + 1e-8))
    return -(t/(y + 1e-8))

# ******************************************End of Losses.py *************************************************************************#