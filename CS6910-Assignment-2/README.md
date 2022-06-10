## CS6910-Assignment-2

### Contributors: Varun Gumma (CS21M070) and Hanumantappa Budihal (CS21M022)

This is the second assignment of CS6910 (Fundamentals of Deep Learning, taught by Prof. Mitesh Khapra) in which we attempt to achieve following goals.
    
- Train a CNN model from scratch and learn how to tune the hyperparameters and visualise filters - Part A
- Finetune a pre-trained model just as you would do in many real world applications - Part B
- Use an existing pre-trained model for a cool application. (TODO : Mention the application name) - Part C

#### 1.Dataset : [iNaturalist](https://www.inaturalist.org/) dataset
This network will be trained and tested using the iNaturalist.Implemented a CNN using keras and trained it on the inaturalist dataset in Part A and fine-tuned pretrained models  to run on the same dataset in Part B

#### 2.Code Structure 
    
    Part A
        |
        +---- best_model.h5 : best performing model from wandb
        |
        +---- guided_backprop_filter_visualization.ipynb: helps us test the best model and visualize filters, feature-maps, predictions and guided backpropogation
        |
        +---- train_wandb.py : helps us test custom configurations and run wandb sweeps.

    Part B
        |
        +---- train_wandb_transfer_learning.py :
        |
        +---- transfer_learning_evaluation.ipynb :

    Part C
        |
        +---- implementation.ipynb: used the YoloV3 model to detect wildlife animal 
        |
        +---- details.txt:  Dataset, model and demo video information

#### 3.Requirements
All the python libraries required to run the program are listed in `requirements.txt`    
Use `$pip install -r requirements.txt` to install the dependencies. **(Use python 3.10 or lower)**

#### 4.Assignment report 
Wandb Report : https://wandb.ai/cs21m070_cs21m022/IITM-CS6910-Projects-CNN/reports/CS6910-Assignment-2--VmlldzoxNzM0ODQz

**README for each part is located in the respective folder**