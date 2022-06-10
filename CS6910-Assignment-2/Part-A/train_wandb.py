# import statements
import wandb
import argparse
import tensorflow as tf
from wandb.keras import WandbCallback
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adamax
mixed_precision.set_global_policy('mixed_float16')
# using mixed_precision speeds up training on GPUs with compute-compatibility >= 8.0
# it also halves the memory usage as keras intelligently switched between 16-bit and 32-bit precision

# some fixed parameters
# this code assumes that the dataset is in the same directory as the script
BASE_PATH = "./inaturalist_12K"
IMG_SIZE = (300, 300)

# set up the argparse
# --wandb: if this flag is set, wandb sweeps are run
# --num-filters: Number of filters in the first convolution layer
# --filter-size: Kernel size of the convolution filters being used
# --filter-multiplier: Equivalent to filter-organization. Whether to double/half/keep the number of filters same across the layers
# --conv-activation: Activation to be used in the conv-block (defaults to ReLU)
# --dense-activation: Activation to be used to in the dense layer before the output layer
# --num-dense: Number of neurons in the dense layer
# --dropout: Dropout in the dense layer
# --optimizer: Name of the optimizer to be used for training
# --learning-rate: learning rate for the optimizer
# --batch-size: batch size for training
# --epochs: number of epochs for training
# --data-augmentation: if this flag is set, data augmentation is performed on the data
# --batch-normalization: if this flag is set, batch normalization is performed after the convolution and before the activation
parser = argparse.ArgumentParser(description="arguments to build the CNN")
parser.add_argument("-wb", "--wandb", action="store_true")
parser.add_argument("-nf", "--num-filters", type=int, default=32, help="number of filters in first layer")
parser.add_argument("-fs", "--filter-size", type=int, default=3, help="convolution filter size for each layer")
parser.add_argument("-fm", "--filter-multiplier", type=float, default=1, help="filter organization for the model: [doubling, halving, same]")
parser.add_argument("-cact", "--conv-activation", type=str, default="relu", help="activation for conv layer")
parser.add_argument("-dact", "--dense-activation", type=str, default="softplus", help="activation for dense layer")
parser.add_argument("-nd", "--num-dense", type=int, default=128, help="number of dense neurons")
parser.add_argument("-drop", "--dropout", type=float, default=0.25, help="dropout probability for dense layer")
parser.add_argument("-opt", "--optimizer", type=str, default="rmsprop", help="optimizer to train the network")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, help="learning_rate for optimizer")
parser.add_argument("-b", "--batch-size", type=int, default=32, help="batch size for the model")
parser.add_argument("-e", "--epochs", type=int, default=20, help="epochs to train the model")
parser.add_argument("-da", "--data-augmentation", action="store_true")
parser.add_argument("-bn", "--batch-normalization", action="store_true")
args = parser.parse_args()

# a function for reading data from the given directory structure
# we put aside 10% data for validation and  rescale our image pixel values by 1/255
# returns the train and validation generators which can be passed to the model.fit() method
def get_data_generators(data_augmentation=True, img_size=IMG_SIZE, batch_size=32):
    if data_augmentation:
        # the following augmentation techniques are used
        data = ImageDataGenerator(rescale=1/255,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  validation_split=0.1,
                                  shear_range=0.25,
                                  zoom_range=[0.25, 1.25],
                                  width_shift_range=0.25,
                                  height_shift_range=0.25,
                                  horizontal_flip=True,
                                  rotation_range=60)
    else:
        data = ImageDataGenerator(rescale=1/255,
                                  validation_split=0.1)

    # here the class_mode is specified as sparse
    # this means the targets are specified as whole numbers (ex. 0, 1, 2 etc) instead of one-hot vectors
    # it is bit memory efficient this way
    train_gen = data.flow_from_directory(f"{BASE_PATH}/train",
                                         target_size=img_size,
                                         batch_size=batch_size,
                                         color_mode="rgb",
                                         class_mode="sparse",
                                         shuffle=True,
                                         seed=123,
                                         subset="training")

    validation_gen = data.flow_from_directory(f"{BASE_PATH}/train",
                                              target_size=img_size,
                                              color_mode="rgb",
                                              class_mode="sparse",
                                              shuffle=True,
                                              seed=123,
                                              subset="validation")
                                              
    return train_gen, validation_gen

# a function for building and running the model as per cmdline arguments
def build_and_run_model_with_cmd_args():
    # construct the optimizer
    optimizer = {
        "sgd": SGD(learning_rate=args.learning_rate, momentum=0.9, nesterov=True),
        "adam": Adam(learning_rate=args.learning_rate),
        "adamax": Adamax(learning_rate=args.learning_rate),
        "rmsprop": RMSprop(learning_rate=args.learning_rate),
    }[args.optimizer]

    # get the train and validation data
    train_gen, validation_gen = get_data_generators(data_augmentation=args.data_augmentation, 
                                                    batch_size=args.batch_size)
    
    # build the model
    model = Sequential()
    # do not forget to add input layer
    # it helps the model know itself
    # also helps us save and re-load the model easily
    model.add(Input(shape=(*IMG_SIZE, 3)))
    # add 5 con-blocks as specified in the question
    for i in range(5):
        model.add(Conv2D(filters=args.num_filters*(args.filter_multiplier ** i),
                         kernel_size=(args.filter_size, args.filter_size)))
        if args.batch_normalization:
            model.add(BatchNormalization())
        model.add(Activation(args.conv_activation))
        # we fix the max-pooling kernel size as 2
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatten the output of the convolution operations
    model.add(Flatten())
    # pass it through a dense layer with some dropout
    model.add(Dense(args.num_dense, activation=args.dense_activation))
    model.add(Dropout(args.dropout))
    model.add(Dense(10, activation="softmax", dtype="float32"))

    # compile and run the model with early-stopping
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # we stop the training if the val_accuracy drops (by 1e-3 or higher) continuously for 4 times                  
    early_stop = EarlyStopping(monitor='val_accuracy', 
                               min_delta=1e-3, 
                               patience=4, 
                               restore_best_weights=True)
    # fit the model to data
    model.fit(train_gen, 
              epochs=args.epochs, 
              validation_data=validation_gen, 
              callbacks=[early_stop])

    model.save("best_model.h5")

#---------------------------------------------------------------------------------------------------------------------#

# a function to run sweeps using wandb
def build_and_run_model_with_wandb(config=None):
    run = wandb.init(config=config)
    config = wandb.config
    # construct the optimizer
    optimizer = {
        "sgd": SGD(learning_rate=config.lr, momentum=0.9, nesterov=True),
        "adam": Adam(learning_rate=config.lr),
        "adamax": Adamax(learning_rate=config.lr),
        "rmsprop": RMSprop(learning_rate=config.lr),
    }[config.optimizer]
    
    # set the name of run
    wandb.run.name = (
        f"num-filters_{config.num_filters}"
        f"_conv-filter-size_{config.filter_size}"
        f"_filter-multiplier_{config.filter_multiplier}"
        f"_data-augmentation_{config.data_augmentation}"
        f"_dropout_{config.dropout}"
        f"_batchnormalization_{config.batch_normalization}"
        f"_optimizer_{config.optimizer}"
        f"_lr_{config.lr}"
        f"_batch-size_{config.batch_size}"
        f"_dense-units_{config.num_dense}"
        f"_dense-activation_{config.dense_activation}"
    )

    # get the train and validation data
    train_gen, validation_gen = get_data_generators(data_augmentation=config.data_augmentation, 
                                                    batch_size=config.batch_size)

    # build the model
    model = Sequential()
    # do not forget to add input layer
    # it helps the model know itself
    # also helps us save and re-load the model easily
    model.add(Input(shape=(*IMG_SIZE, 3)))
    for i in range(5):
        model.add(Conv2D(filters=config.num_filters*(config.filter_multiplier ** i),
                         kernel_size=(config.filter_size, config.filter_size)))
        if config.batch_normalization:
            model.add(BatchNormalization())
        # here we only use relu
        # other activations make it difficult to run guided-backprop
        model.add(Activation("relu"))
        # we fix the max-pooling kernel size as 2
        model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatten the output of the convolution operations
    model.add(Flatten())
    # pass it through a dense layer with some dropout
    model.add(Dense(config.num_dense, activation=config.dense_activation))
    model.add(Dropout(config.dropout))
    model.add(Dense(10, activation="softmax", dtype="float32"))

    # compile and run the model with early-stopping
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    
    # we stop the training if the val_accuracy drops (by 1e-3 or higher) continuously for 4 times
    # wandbcallback helps log the training to wandb
    early_stop = EarlyStopping(monitor='val_accuracy',
                               min_delta=1e-3,
                               patience=4,
                               restore_best_weights=True)
    wandb_callback = WandbCallback(monitor='val_accuracy',
                                   generator=validation_gen)
    
    # fit the model to data 
    model.fit(train_gen, 
              epochs=25,
              validation_data=validation_gen, 
              callbacks=[early_stop, wandb_callback])

    # save the model as a .h5 file
    model.save(f"{wandb.run.name}.h5")
    run.finish()

# driver code
# run the required function according to wandb flag
if not args.wandb:
    build_and_run_model_with_cmd_args()
else:
    # setup the sweep config
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'values': [1e-3, 1e-4, 7e-5]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'num_dense': {
                'values': [64, 128, 256, 512]
            },
            'dense_activation': {
                'values': ['selu', 'softplus', 'relu', 'leaky_relu'],
            },
            'optimizer':{
                'values': ['sgd', 'adamax', 'rmsprop', 'adam']
            },
            'batch_normalization': {
                'values': [True, False]
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'data_augmentation': {
                'values': [True, False]
            },
            'num_filters': {
                'values': [16, 32, 64],
            },
            'filter_multiplier': {
                'values': [0.5, 1, 2],
            },
            'filter_size': {
                'values': [3, 5],
            }
        }
    }

    # call the sweep
    sweep_id = wandb.sweep(sweep_config, project='IITM-CS6910-Projects-CNN')
    wandb.agent(sweep_id, function=build_and_run_model_with_wandb)