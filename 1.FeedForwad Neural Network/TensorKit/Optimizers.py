# ***************************************************************************************************************************
# File      : Optimizers.py 
# Purpose   : Implemeted the collection of commonly used optimizers 
# Author    : Varun Gumma & Hanumantappa Budihal
#
# Date      : 08-02-2022 
# Bugs      : NA
# Change Log: 
# ****************************************************************************************************************************/

import numpy as np 

class SGD:
    def __init__(self, lr=1e-3, momentum=0, nesterov=False):
        self.vW = []
        self.vB = []
        self.lr = lr
        self.name = "sgd"
        self.momentum = momentum
        self.nesterov = nesterov

    def watch(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        for layer in layers:
            self.vW.append(np.zeros(layer.W.shape))
            self.vB.append(np.zeros(layer.B.shape))

    def look_ahead(self):
        for i in range(self.n_layers):
            self.layers[i].W -= self.momentum * self.vW[i] 
            self.layers[i].B -= self.momentum * self.vB[i]

    def look_back(self):
        for i in range(self.n_layers):
            self.layers[i].W += self.momentum * self.vW[i] 
            self.layers[i].B += self.momentum * self.vB[i]

    def step(self):
        for i in range(self.n_layers):
            self.vW[i] = (self.momentum * self.vW[i]) + (self.lr * self.layers[i].dW)
            self.vB[i] = (self.momentum * self.vB[i]) + (self.lr * self.layers[i].dB)
            self.layers[i].W -= self.vW[i]
            self.layers[i].B -= self.vB[i]

#--------------------------------------------------------------------------------------------------------------------------------#

class Adagrad:
    def __init__(self, lr=1e-3, eps=1e-8):
        self.vW = []
        self.vB = []
        self.lr = lr
        self.eps = eps
        self.name = "adagrad"

    def watch(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        for layer in layers:
            self.vW.append(np.zeros(layer.W.shape))
            self.vB.append(np.zeros(layer.B.shape))

    def step(self):
        for i in range(self.n_layers):
            self.vW[i] += np.square(self.layers[i].dW)
            self.vB[i] += np.square(self.layers[i].dB)
            self.layers[i].W -= (self.lr/(np.sqrt(self.vW[i] + self.eps))) * self.layers[i].dW
            self.layers[i].B -= (self.lr/(np.sqrt(self.vB[i] + self.eps))) * self.layers[i].dB

#---------------------------------------------------------------------------------------------------------------------------#

class RMSprop:
    def __init__(self, lr=1e-3, beta=0.9, eps=1e-8):
        self.vW = []
        self.vB = []
        self.lr = lr
        self.eps = eps
        self.beta = beta
        self.name = "rmsprop"

    def watch(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        for layer in layers:
            self.vW.append(np.zeros(layer.W.shape))
            self.vB.append(np.zeros(layer.B.shape))

    def step(self):        
        for i in range(self.n_layers):
            self.vW[i] = (self.beta * self.vW[i]) + ((1 - self.beta) * np.square(self.layers[i].dW))
            self.vB[i] = (self.beta * self.vB[i]) + ((1 - self.beta) * np.square(self.layers[i].dB))
            self.layers[i].W -= (self.lr/(np.sqrt(self.vW[i] + self.eps))) * self.layers[i].dW
            self.layers[i].B -= (self.lr/(np.sqrt(self.vB[i] + self.eps))) * self.layers[i].dB

#---------------------------------------------------------------------------------------------------------------------------#

class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t = 0
        self.vW = []
        self.vB = []
        self.mW = []
        self.mB = []
        self.lr = lr
        self.eps = eps
        self.name = "adam"
        self.beta1 = beta1
        self.beta2 = beta2

    def watch(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        for layer in layers:
            self.vW.append(np.zeros(layer.W.shape))
            self.vB.append(np.zeros(layer.B.shape))
            self.mW.append(np.zeros(layer.W.shape))
            self.mB.append(np.zeros(layer.B.shape))

    def step(self):
        self.t += 1
        beta1_t = self.beta1 ** self.t 
        beta2_t = self.beta2 ** self.t
        for i in range(self.n_layers):
            self.mW[i] = (self.beta1 * self.mW[i]) + ((1 - self.beta1) * self.layers[i].dW)
            self.mB[i] = (self.beta1 * self.mB[i]) + ((1 - self.beta1) * self.layers[i].dB)
            self.vW[i] = (self.beta2 * self.vW[i]) + ((1 - self.beta2) * np.square(self.layers[i].dW))
            self.vB[i] = (self.beta2 * self.vB[i]) + ((1 - self.beta2) * np.square(self.layers[i].dB))

            mhatW = self.mW[i]/(1 - beta1_t)
            mhatB = self.mB[i]/(1 - beta1_t)
            vhatW = self.vW[i]/(1 - beta2_t)
            vhatB = self.vB[i]/(1 - beta2_t)

            self.layers[i].W -= (self.lr/(np.sqrt(vhatW + self.eps))) * mhatW
            self.layers[i].B -= (self.lr/(np.sqrt(vhatB + self.eps))) * mhatB

#---------------------------------------------------------------------------------------------------------------------------#

class Nadam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t = 0
        self.vW = []
        self.vB = []
        self.mW = []
        self.mB = []
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.name = "nadam"

    def watch(self, layers):
        self.layers = layers
        self.n_layers = len(layers)
        for layer in layers:
            self.vW.append(np.zeros(layer.W.shape))
            self.vB.append(np.zeros(layer.B.shape))
            self.mW.append(np.zeros(layer.W.shape))
            self.mB.append(np.zeros(layer.B.shape))

    def step(self):
        self.t += 1
        beta1_t = self.beta1 ** self.t 
        beta2_t = self.beta2 ** self.t
        for i in range(self.n_layers):
            self.mW[i] = (self.beta1 * self.mW[i]) + ((1 - self.beta1) * self.layers[i].dW)
            self.mB[i] = (self.beta1 * self.mB[i]) + ((1 - self.beta1) * self.layers[i].dB)
            self.vW[i] = (self.beta2 * self.vW[i]) + ((1 - self.beta2) * np.square(self.layers[i].dW))
            self.vB[i] = (self.beta2 * self.vB[i]) + ((1 - self.beta2) * np.square(self.layers[i].dB))

            mhatW = self.mW[i]/(1 - beta1_t)
            mhatB = self.mB[i]/(1 - beta1_t)
            vhatW = self.vW[i]/(1 - beta2_t)
            vhatB = self.vB[i]/(1 - beta2_t)

            dW = (self.beta1 * mhatW) + (((1 - self.beta1)/(1 - beta1_t)) * self.layers[i].dW)
            dB = (self.beta1 * mhatB) + (((1 - self.beta1)/(1 - beta1_t)) * self.layers[i].dB)
            self.layers[i].W -= (self.lr/(np.sqrt(vhatW + self.eps))) * dW
            self.layers[i].B -= (self.lr/(np.sqrt(vhatB + self.eps))) * dB

#---------------------------------------------------------------------------------------------------------------------------#