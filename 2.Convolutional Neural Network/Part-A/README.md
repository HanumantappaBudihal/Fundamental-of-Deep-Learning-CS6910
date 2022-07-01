# CS6910-Assignment-2 Part-A

This part has two files, namely ```train_wandb.py``` which helps us train custom configurations or run wandb sweeps, and ```guided_backprop_filter_visualization.ipynb``` which helps us test the best model and visualize filters, feature-maps, predictions and guided backpropogation. Both these files assume the dataset is present in the same directory with the name ```inaturalist_12K```.

Before training, all images are resized to (300, 300) using the inbuilt ```nearest``` strategy (in ```from_from_directory```)

## a) train_wandb.py
This file supports the following command line arguments:

```
--wandb: if this flag is set, wandb sweeps are run
--num-filters: Number of filters in the first convolution layer
--filter-size: Kernel size of the convolution filters being used
--filter-multiplier: Equivalent to filter-organization. Whether to double/half/keep the number of filters same across the layers
--conv-activation: Activation to be used in the conv-block (defaults to ReLU)
--dense-activation: Activation to be used to in the dense layer before the output layer
--num-dense: Number of neurons in the dense layer
--dropout: Dropout in the dense layer
--optimizer: Name of the optimizer to be used for training
--learning-rate: learning rate for the optimizer. Should one out of sgd, rmsprop, adam, adamax
--batch-size: batch size for training
--epochs: number of epochs for training
--data-augmentation: if this flag is set, data augmentation is performed on the data
--batch-normalization: if this flag is set, batch normalization is performed after the convolution and before the activation
```
An example on how to run sweeps:
```
python train_wandb.py --wandb
```
The sweeps are run using ```Bayes Strategy``` and ```EarlyStopping```, i.e. the training is terminated if the validation_accuracy of the model drops by 0.001 (or higher) for four consecutive epochs. 

After the sweeps are completed, it generates a number of ```.h5``` saved files in the same directory and it contains the model and its trained parameters.

The names of these files will be the same the run-name in wandb. Choose the best performing model from wandb, and rename the corresponding saved as ```best_model.h5```. 

An example on how to train a custom configuration:
```
python train_wandb.py --num-filters=32 \ 
                      --filter-size=3 \
                      --filter-multiplier=1 \
                      --conv-activation="swish" \
                      --dense-activation="softplus" \
                      --num-dense=512 \
                      --dropout=0.2 \
                      --optimizer="rmsprop"
                      --learning-rate=0.001 \
                      --batch-size=64 \
                      --epochs=20 \
                      --data-augmentation \
                      --batch-normalization
```
After the custom configuration model is trained, the model and its trained parameters are saved. But this time, directly a ```best_model.h5``` file is generated.

## b) guided_backprop_filter_visualization.ipynb
Once we have the ```best_model.h5``` file in place, we can run the ```guided_backprop_filter_visualization``` jupyter notebook (using ```Run All Cells```). 

It generates all the required plots of predictions, filters, feature-maps and guided-backpropogation. All these plots are saved to the current directory and can be viewed in any image-viewer. 

**Note**: The images generated here may be different from the images in the report as random test-images are chosen everytime. Also, the test dataset is used only in this part and it is not even read while training/sweeps.
