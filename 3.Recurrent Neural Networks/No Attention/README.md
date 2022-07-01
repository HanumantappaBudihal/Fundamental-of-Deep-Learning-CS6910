# Vanilla Model
We have 3 files in this directory, namely, ```VanillaModels.py```, ```train_vanilla_seq2seq.py``` and ```test_vanilla_seq2seq.ipynb```. To run the code, make sure the ```dakshina``` dataset is present in this folder. 

The ```VanillaModels.py``` contains the definition of the ```Encoder``` and ```Decoder``` modules. These two are put togther to form the ```Transliteration``` model. The transliteration module has custom ```train_step``` and ```test_step``` defined which will help run with and without *teacher-forcing* during training and inference (validation/testing) respectively.

For training the transliteration model, the following flags can be used: 
```
--wandb: if this flag is set, then wandb sweeps are run, else the custom hyperparameter configuration is used
--recurrent-layer: type of layer for encoder and decoder, should one of [rnn/gru/lstm]
--num-recurrent-units: number of cells in the recurrent layer, will be same for encoder and decoder as they need to have same cell_state_dim
--enc-embedding-size: embedding dimension for encoder
--dec-embedding-size: embedding dimension for decoder
--dropout: dropout for both encoder and decoder
--num-enc-recurrent-layers: number of recurrent layers in the encoder
--num-dec-recurrent-layers: number of recurrent layers in the decoder
--learning-rate: learning rate for the optimizer
--weight-decay: decoupled weight-decay for the optimizer
--optimizer: optimizer to be used for training, should be one of [adamw/ranger]
--batch-size: batch size for the model
```

For example, to train the model on a custom hyperparameter configuration, run the following command:
```
python train_vanilla_seq2seq.py --recurrent-layer="gru" \
                                --num-recurrent-units=1024 \
                                --enc-embedding-size=128 \
                                --dec-embedding-size=128 \
                                --dropout=0.2 \
                                --num-enc-recurrent-layers=3 \
                                --num-dec-recurrent-layers=1 \
                                --learning-rate=0.001 \
                                --weight-decay=0.00001 \
                                --optimizer="sgdw" \
                                --batch-size=1024 
```

and, to train the model with wandb and visualize the sweeps, run the following command:
```
python train_vanilla_seq2seq.py --wandb
```

After the sweeps are completed, lookup the best configuration from WandB and re-train the architecture with the best hyperparameter set. This needs to be done as model can't be saved due to custom objects (```Encoder``` and ```Decoder```) being defined. Note that, while training, the test dataset it not even read and the best model is selected based on the validation word-level accuracy.

To re-train the best model and obtain test word-level accuracy along prediction visualization, run all cells of the ```test_vanilla_seq2seq.ipynb``` notebook. Only in this program, we use the test dataset.

The code for prediction visualization selects random data from the test dataset for plots. Hence, the words shown in the report and the ones you get when you run the code may differ.