## CS6910-Assignment-3

### Contributors: Varun Gumma (CS21M070) and Hanumantappa Budihal (CS21M022)

This is the third assignment of CS6910 (Fundamentals of Deep Learning, taught by Prof. Mitesh Khapra) in which we attempt to achieve following goals.
    
- Learn how to model sequence to sequence learning problems using Recurrent Neural Networks 
- Compare different cells such as vanilla RNN, LSTM and GRU
- understand how attention networks overcome the limitations of vanilla seq2seq models
- Visualise the interactions between different components in a RNN based model
- Finetune the GPT2 model to generate lyrics for an English song.

#### 1.Dataset : [Dakshina dataset ](https://github.com/google-research-datasets/dakshina) dataset

This dataset is a collection of text in both Latin and Native scripts for the 12 south Asian languages.
#### 2.Code Structure 
    
    No Attention
                |
                +---- test_vanilla_seq2seq.ipynb 
                |
                +---- train_vanilla_seq2seq.ipynb
                |
                +---- VanillaModels.py 

    Attention
            |
            +---- AttentionModels.py 
            |
            +---- test_attention_seq2seq.ipynb 
            |
            +---- train_attention_seq2seq.py

    GPT  
        |
        +---- final_song.txt
        |
        +---- rihanna.txt
        |
        +---- run_clm.py
        |
        +---- songs_merged.txt
        |
        +---- train_GPT2.ipynb

#### 3.Requirements
All the python libraries required to run the program are listed in `requirements.txt`    
Use `$pip install -r requirements.txt` to install the dependencies. **(Use python 3.10 or lower)**

#### 4.Assignment report 
Wandb Report : https://wandb.ai/cs21m070_cs21m022/IITM-CS6910-Projects-Seq2Seq/reports/CS6910-Assignment-3--VmlldzoxODc1MTQ3?accessToken=f7sk81xtdrjyxmr91oxgvz7u92ldkimzf3j9ei2xbbbgamjr7vtjn15bmqgk6m3s

**README for each part is located in the respective folder**
