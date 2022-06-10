# CS6910-Assignment-2 Part-B

This part has two files, namely ```train_wandb_transfer_learning.py``` which helps us train custom configurations or run wandb sweeps, and ```transfer_learning_evaluation.ipynb``` which helps us to evaluate the pre-trained-model. 

Both these files assume the dataset is present in the same directory with the name ```inaturalist_12K```. Before training, all images are resized to (224, 224) using the inbuilt ```nearest``` strategy (in ```from_from_directory```)

## a) train_wandb_transfer_learning.py
This file supports the following command line arguments:

```
--wandb: if this flag is set, wandb sweeps are run
--base-model: a name of the pre-trained model to be used. Has to be one of xception, resnet50, inceptionV3, inception_resnetV2
--batch-size: batch_size for training
--learning-rate: learning_rate for the optimizer
--weight-decay: decoupled weight_decay for the optimizer
--freeze: fraction of inital layers to freeze. Should be a value between [0, 1]
--data-augmentation: if this flag is set, data_augmentation is performed in the data 
```
An example on how to run sweeps:
```
python train_wandb_transfer_learning.py --wandb
```

An example on how to train a custom configuration:
```
python train_wandb_transfer_learning.py --base-model="xception" \ 
                                        --batch-size=16 \
                                        --learning-rate=0.001 \
                                        --weight-decay=0.001 \
                                        --freeze=0.33 \
                                        --data-augmentation \
```
After the sweeps are completed, lookup the best hyperparameter set from wandb.

## b) transfer_learning_evaluation.ipynb
Once the best hyperparameter set is identified from wandb, we re-train the model with those hyperparameter set in this juputer notebook. You can just use ```Run All Cells``` to excute the whole code and get the test evaluation metrics at the end. Also, the test dataset is used only in this part and it is not even read while training/sweeps.
