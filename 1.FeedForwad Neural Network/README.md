## CS6910-Assignment-1

### Contributors: Varun Gumma (CS21M070) and Hanumantappa Budihal (CS21M022)

This is the first assignment of CS6910 (Fundamentals of Deep Learning, taught by Prof. Mitesh Khapra) in which we attempt to create a FeedForwad Neural Network from scratch using only NumPy.

#### 1.Dataset : [fashion mnsit](https://keras.io/api/datasets/fashion_mnist/)  
This network will be trained and tested using the Fashion-MNIST dataset. Specifically, given an input image (28 x 28 = 784 pixels) from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes.

#### 2.Code Structure 
    
    TensorKit
        |
        +---- Activations.py: Contains commonly used activation functions [with dervatives]
        |
        +---- Losses.py: Contains commonly used loss functions [with derivatives]
        |
        +---- Layers.py: Dense layer which performs f(Wx + B) and stores activations, pre-activations and gradients
        |
        +---- Optimizers.py: Implemented  SGD (momentum, nesterov), Adagrad, RMSprop, Adam and Nadam Optimizers
        |
        +---- Models.py: Sequential architecture which uses all the above classes to build a concrete DNN

    Q1.ipynb - Solution for question 1 
    Q4.ipynb - Solution for question 4
    Q7.ipynb - Solution for question 7
    Q8.ipynb - Solution for question 8
    Q10.ipynb - Solution for question 10
  
#### 3.Requirements
All the python libraries required to run the program are listed in `requirements.txt`    
Use `$pip install -r requirements.txt` to install the dependencies. **(Use python 3.10 or lower)**

#### 4.How to run the code

The structure of the each notebook [Q1,Q4,Q7,Q8,Q10] is such that it can be run sequentially till the end just by pressing ```Shift + Enter``` for each cell or pressing ```Run all cells```.

#### 5.Implementation 
1. For creating a model:

    ```python
    # define the model architecture
    model = Sequential()

    model.add(Dense(trainX.shape[1], hn, activation=activation, initialization=w_init))
    for _ in range(hl):
        model.add(Dense(hn, hn, activation=activation, initialization=w_init))
    
    model.add(Dense(hn, trainy.shape[1], activation="softmax", initialization=w_init))
    ```

2. To compile the model so that it knows what loss and optimizer it's working with, use the following syntax (here ```optimizer``` is a Optimizer class object)

    ```python
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", weight_decay=l2)
    ```

3. To train the model, you have to specify both training and validation datasets along with batch size and number of epochs to train for in the given syntax

    ```python
    model.fit(trainX, trainy, batch_size=bs, epochs=epochs, validation_data=(valX, valy))
    ```

4. To evaluate the model on a separate test dataset, use the following code 

    ```python
    loss, acc = model.evaluate(testX, testy, batch_size=bs)
    ```

5. For running a wandb sweep,  use the ```build_and_run_model()``` function for setting up wandb sweeps using our framework. Set up a wandb sweep as follows and run the sweep using the commands shown.

    ```python

    # setting up hyperparameter configurations for the sweep
    sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid', 'leaky_relu']
        },
        'lr': {
            'values': [1e-2, 1e-3, 1e-4]
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'hidden_layer_size': {
            'values': [32, 64, 128]
        },
        'epochs': {
            'values': [5, 10, 15]
        },
        'num_hidden_layers': {
            'values': [3, 4, 5]
        },
        'optimizer': {
            'values': ['sgd', 'mgd', 'nag', 'adagrad', 'rmsprop', 'adam', 'nadam']
        },
        'l2': {
            'values': [0, 5e-2, 5e-3]
        },
        'weight_init': {
            'values': ['xavier', 'random', 'normal']
        }
    }}

    # Login to wandb
    wandb.login()
    
    # setting up the sweep        
    sweep_id = wandb.sweep(sweep_config, project='IITM-CS6910-Projects')

    # running the sweep
    wandb.agent(sweep_id, function=build_and_run_model)

    ```
#### 6.Utility library for neural network (TensorKit)

1. Activation functions

    ```python
    identity()
    sigmoid()
    tanh()
    ReLU()
    leaky_ReLU()
    softmax()

    ```

2. Loss functions

    ```python
    MeanSquaredLoss()
    BinaryCrossEntropy()
    CategoricalCrossEntropy()

    ```

3. Optimizers 

    ```python
    SGD()
    Adagrad()
    RMSprop()
    Adam()
    Nadam()

    ```

4. Layers

    ```python
    Dense() # a fully-connected layer
    ```
5. Models

    ```python
    Sequential() # Sequential model is appropriate for a plain stack of layers
    ```
#### 7.Assignment report 
Wandb Report : https://wandb.ai/cs21m070_cs21m022/IITM-CS6910-Projects/reports/CS6910-Assignment-1--VmlldzoxNTM0ODEw
