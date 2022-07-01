# ***************************************************************************************************************************
# File      : Models.py 
# Purpose   : Build models 
# Author    : Varun Gumma & Hanumantappa Budihal
#
# Date      : 10-02-2022 
# Bugs      : NA
# Change Log: 
#            ->23-02-2022 : Added the function comments to all methods.
# ****************************************************************************************************************************/

import wandb
import numpy as np 
from .Losses import *

# a short lambda function to later help compute accuracy
accuracy_score = lambda y, t : np.mean(np.equal(y.argmax(axis=1), t.argmax(axis=1)))

class Sequential:
    def __init__(self):
        self.layers = []
        self.loss_f = None
        self.weight_decay = 0
        self.optimizer = None
        self.isAccelerated = None
        

    def add(self, layer):
        """  
        adds a new layer to the network

        Parameters
	    ----------
	    layer : an instance of the layer class

        Returns
	    -------
        NA, just adds the layer to the model     
        """
        self.layers.append(layer)


    def compile(self, optimizer, loss, weight_decay=0):
        """  
        compiles the model and links the optimizer

        Parameters
	    ----------
	    optimizer : an instance of the layer class
        loss : 
        weight_decay : 
        Returns
	    -------
        NA, just adds the layer to the model     
        """
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.optimizer.watch(self.layers)
        self.isAccelerated = (self.optimizer.name == "sgd" \
                              and self.optimizer.nesterov)
        self.loss_f = {
            "mean_squared_error": MeanSquaredLoss,
            "binary_crossentropy": BinaryCrossEntropy,
            "categorical_crossentropy": CategoricalCrossEntropy
        }[loss]


    def forward(self, X):
        """  
        forward propagates data   
        """
        for layer in self.layers:
            X = layer(X)
        return X


    def backward(self, X, y, t):
        """  
        back propagates "errors"

        Parameters
	    ----------
	    X : input data
        y : predictions
        t : target variables

        Returns
	    -------
        NA, just sets the gradients of the paramaters according to chain-rule     
        """
        dL = self.loss_f(y, t, derivative=True)
        for i in reversed(range(len(self.layers))):
            cache_z = (X if not i else self.layers[i-1].Z)
            dL = self.layers[i].backward(dL, cache_z, self.weight_decay)


    def fit(self, X, Y, batch_size=2, epochs=1, validation_data=None, wandb_log=False):
        """  
        "fits" the data in batches by simultaneous forwardprop and backprop

        Parameters
	    ----------
        X : input data
        y : target variables
        batch_size : batch_size
        epochs : no. epochs to run the model
        validation_data : tuple of (X_val, Y_val)
        wandb_log : boolean flag to print to console or log to wandb

        Returns
	    -------
        NA, just prints/logs the loss, accuracy, val_loss and val_accuracy     
        """
        N = X.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(N)
            X, Y = X[perm], Y[perm]
            
            for i in range(0, N, batch_size): 
                j = min(i + batch_size, N)
                x, t = X[i : j], Y[i : j]

                if self.isAccelerated:
                    self.optimizer.look_ahead()

                y = self.forward(x)
                self.backward(x, y, t)
                
                if self.isAccelerated:
                    self.optimizer.look_back()
                self.optimizer.step()

            y = self.predict(X, batch_size)
            loss, acc = self.evaluate(X, Y, batch_size, mode="train")
            desc = f"epoch {epoch+1}/{epochs}:\ttrain_loss: {loss:.4f}, train_accuracy: {acc:.4f}"
            if validation_data is not None:
                val_loss, val_acc = self.evaluate(*validation_data, batch_size)
                desc += f", val_loss: {val_loss:.4f}, val_accuracy: {val_acc:.4f}"
                if wandb_log:
                    wandb.log({
                        "epoch": epoch+1,
                        "train_loss": loss,
                        "train_accuracy": acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc
                    })
                print(desc)
        print("\n")
            

    def predict(self, X, batch_size=1, predict_proba=True):
        """          
        TODO : 
        Parameters
	    ----------
        X :
        batch_size : 
        predict_proba :

        Returns
	    -------
            
        """
        N, preds = X.shape[0], []
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            y = self.forward(X[i : j])
            preds.extend(y if predict_proba else y.argmax(axis=1))
        return np.array(preds) 


    def evaluate(self, X, Y, batch_size=1, mode="eval"):
        """          
        TODO : 
        Parameters
	    ----------
        X :
        batch_size : 
        predict_proba :

        Returns
	    -------
            
        """
        y = self.predict(X, batch_size)
        acc = accuracy_score(y, Y)
        loss = self.loss_f(y, Y)
        if mode == "train":
            loss += (0.5 * self.weight_decay * self.L2_loss())
        return loss/len(X), acc

    def L2_loss(self):
        """          
        L2 Loss Function is used to minimize the error which is the sum of the all the
        squared differences between the true value and the predicted value

        """
        return sum([np.sum(np.square(layer.W)) for layer in self.layers])