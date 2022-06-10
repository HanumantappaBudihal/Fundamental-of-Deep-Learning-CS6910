# import statements
import wandb
import argparse
from math import ceil
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception

# some fixed parameters
# this code assumes that the dataset is in the same directory as the script
# we fix the epochs as 5, as it takes really long for training
BASE_PATH = "./inaturalist_12K"
IMG_SIZE = (224, 224)
EPOCHS = 5

# set up the argparse
# --wandb: if this flag is set, wandb sweeps are run
# --base-model: a name of the pre-trained model to be used. Has to be one of [xception, resnet50, inceptionV3, inception_resnetV2]
# --batch-size: batch_size for training
# --learning-rate: learning_rate for the optimizer
# --weight-decay: decoupled weight_decay for the optimizer
# --freeze: fraction of inital layers to freeze. Should be a value between [0, 1]
# --data-augmentation: if this flag is set, data_augmentation is performed in the data 
parser = argparse.ArgumentParser(description="arguments to build the pre-trained-model")
parser.add_argument("-wb", "--wandb", action="store_true", help="flag to run wandb sweeps for a specific configuration")
parser.add_argument("-bm", "--base-model", type=str, default="resnet50", help="name of the pre-trained model to be used. Has to be one of [xception, resnet50, inceptionV3, inception_resnetV2]")
parser.add_argument("-bs", "--batch-size", type=int, default=64, help="batch-size for the model")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2, help="learning rate for the RANGER optimizer")
parser.add_argument("-wd", "--weight-decay", type=float, default=1e-3, help="decoupled weight-decay for the RANGER optimizer")
parser.add_argument("-f", "--freeze", type=float, default=0, help="fraction of layers to freeze")
parser.add_argument("-da", "--data-augmentation", action="store_true", help="flag to perform data-augmentation or not")
args = parser.parse_args()

# a function for reading data from the given directory structure
# we put aside 10% data for validation and  rescale our image pixel values by 1/255
# returns the train and validation generators which can be passed to the model.fit() method
def get_data_generators(data_augmentation=True, batch_size=32, img_size=IMG_SIZE):
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
                                              batch_size=batch_size,
                                              color_mode="rgb",
                                              class_mode="sparse",
                                              shuffle=True,
                                              seed=123,
                                              subset="validation")
                                              
    return train_gen, validation_gen


# a function for building and running the model as per cmdline arguments
def build_and_run_model_with_cmd_args():
    # get the train and validation data
    train_gen, validation_gen = get_data_generators(args.data_augmentation, args.batch_size)
    # construct the model
    # it may take a while as the weights have to downloaded
    if args.base_model == "xception":
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif args.base_model == "resnet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif args.base_model == "inceptionV3":
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif args.base_model == "inception_resnetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    
    # freeze the specified fraction of layers
    # by setting their trainable variable to False
    # by default, all layers have trainable set as True
    N = len(base_model.layers)
    for layer in base_model.layers[: ceil(args.freeze*N)]:
        layer.trainable = False

    # build the model
    # add all necessary layers
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="swish"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))

    # we use a new optimizer Lookahead + Radam (a.k.a RANGER) with specified learning_rate and weight_decay
    # this optimizer has proved to converge really fast and generalize well
    # given the limited number of epochs we were able to make, we tried to use the most efficient optimizer to get the best results
    OPTIM = Lookahead(RectifiedAdam(learning_rate=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True))
    model.compile(optimizer=OPTIM, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # define the early stopping
    # we stop the training if the val_accuracy drops (by 1e-4 or higher) continuously for 4 times
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=4, restore_best_weights=True)
    # fit the data to the model
    model.fit(train_gen, epochs=EPOCHS, validation_data=validation_gen, callbacks=[early_stop])

#---------------------------------------------------------------------------------------------------------------------#

# a function to run sweeps using wandb
def build_and_run_model_with_wandb(config=None):
    # initiate the run
    run = wandb.init(config=config)
    config = wandb.config
    
    # set the run name
    wandb.run.name = (
        f"base-model_{config.base_model}"
        f"_freeze-fraction_{config.freeze}"
        f"_data-augmentation_{config.data_augmentation}"
        f"_batch-size_{config.batch_size}"
        f"_lr_{config.lr}"
        f"_weight-decay_{config.weight_decay}"
    )
    
    # get the train and validation data
    train_gen, validation_gen = get_data_generators(config.data_augmentation, config.batch_size)
    # construct the model
    # it may take a while as the weights have to downloaded
    if config.base_model == "xception":
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif config.base_model == "resnet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif config.base_model == "inceptionV3":
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    elif config.base_model == "inception_resnetV2":
        base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    
    # freeze the specified fraction of layers
    # by setting their trainable variable to False
    # by default, all layers have trainable set as True
    N = len(base_model.layers)
    for layer in base_model.layers[ : ceil(config.freeze*N)]:
        layer.trainable = False

    # build the model
    # add all necessary layers
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="swish"))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation="softmax"))

    # we use a new optimizer Lookahead + Radam (a.k.a RANGER) with specified learning_rate and weight_decay
    # this optimizer has proved to converge really fast and generalize well
    # given the limited number of epochs we were able to make, we tried to use the most efficient optimizer to get the best results
    OPTIM = Lookahead(RectifiedAdam(learning_rate=config.lr, weight_decay=config.weight_decay, amsgrad=True))
    model.compile(optimizer=OPTIM, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    # define the early stopping
    # we stop the training if the val_accuracy drops (by 1e-4 or higher) continuously for 4 times
    early_stop = EarlyStopping(monitor="val_accuracy", min_delta=1e-4, restore_best_weights=True, patience=4)
    # fit the data to the model and save the training history
    hist = model.fit(train_gen, epochs=EPOCHS, validation_data=validation_gen, callbacks=[early_stop])
    T = hist.history
    
    # log the values to wandb
    # Due to some errors while using WandBCallback, we chose this method
    for (loss, val_loss, accuracy, val_accuracy) in zip(T["loss"], T["val_loss"], T["accuracy"], T["val_accuracy"]):
        wandb.log({
            "part-b-val_loss":  val_loss,
            "part-b-val_accuracy": val_accuracy,
            "part-b-loss": loss,
            "part-b-accuracy": accuracy
        })
    
    # close the run
    run.finish()

#-------------------------------------------------------------------------------------------------------------------#

# driver code
# run the required function according to wandb flag
if not args.wandb:
    build_and_run_model_with_cmd_args()
else:
    # setup the sweep config
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'part-b-val_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'base_model': {
                'values': ["xception", "resnet50", "inceptionV3", "inception_resnetV2"]
            },
            'freeze': {
                'values': [0, 0.33, 0.66, 1]
            },
            'data_augmentation': {
                'values': [True, False]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'lr': {
                'values': [1e-2, 1e-3, 1e-4]
            },
            'weight_decay': {
                'values': [1e-2, 1e-3, 1e-4]
            }
        }
    }

    # call the sweep
    sweep_id = wandb.sweep(sweep_config, project='IITM-CS6910-Projects-CNN')
    wandb.agent(sweep_id, function=build_and_run_model_with_wandb)