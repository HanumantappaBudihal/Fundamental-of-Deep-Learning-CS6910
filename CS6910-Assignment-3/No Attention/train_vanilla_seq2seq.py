# import statements
import wandb
import argparse
import numpy as np
import pandas as pd
from VanillaModels import *
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow_addons.optimizers import RectifiedAdam, Lookahead, NovoGrad, SGDW, AdamW

# define cmdline arguments to train a model with custom hyperparameters
# --wandb: if this flag is set, then wandb sweeps are run, else the custom hyperparameter configuration is used
# --recurrent-layer: type of layer for encoder and decoder, should one of [rnn/gru/lstm]
# --num-recurrent-units: number of cells in the recurrent layer, will be same for encoder and decoder as they need to have same cell_state_dim
# --enc-embedding-size: embedding dimension for encoder
# --dec-embedding-size: embedding dimension for decoder
# --dropout: dropout for both encoder and decoder
# --num-enc-recurrent-layers: number of recurrent layers in the encoder
# --num-dec-recurrent-layers: number of recurrent layers in the decoder
# --learning-rate: learning rate for the optimizer
# --weight-decay: decoupled weight-decay for the optimizer
# --optimizer: optimizer to be used for training, should be one of [sgdw/adamw/ranger/novograd]
# --batch-size: batch size for the model
parser = argparse.ArgumentParser(description="arguments to build the Seq2Seq Model")
parser.add_argument("-wb", "--wandb", action="store_true")
parser.add_argument("-rl", "--recurrent-layer", type=str, default="rnn", help="type of layer for encoder and decoder [rnn/gru/lstm]")
parser.add_argument("-nru", "--num-recurrent-units", type=int, default=256, help="recurrent units for each encoder recurrent layer")
parser.add_argument("-eemb", "--enc-embedding-size", type=int, default=512, help="embedding size of encoder")
parser.add_argument("-demb", "--dec-embedding-size", type=int, default=512, help="embedding size of decoder")
parser.add_argument("-drop", "--dropout", type=float, default=0.25, help="dropout for recurrent units")
parser.add_argument("-nerl", "--num-enc-recurrent-layers", type=int, default=3, help="number of recurrent layers for encoder")
parser.add_argument("-ndrl", "--num-dec-recurrent-layers", type=int, default=3, help="number of recurrent layers for decoder")
parser.add_argument("-lr", "--learning-rate", type=float, default=1e-2, help="learning rate for optimizer")
parser.add_argument("-wd", "--weight-decay", type=float, default=1e-3, help="weight decay for optimizer")
parser.add_argument("-o", "--optimizer", type=str, default="ranger", help="optimizer for training the model [should be one of `ranger`, `adamw`, `novograd`, `sgdw`]")
parser.add_argument("-bs", "--batch-size", type=int, default=256, help="batch size for the model")
args = parser.parse_args()

# set the path the directories
# code assumes that the dakshina folder is the same directory as this code
train_dir = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv"
dev_dir = "./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv"

# define a function to compute word-level accuracy
# after stripping of all pad-tokens
def compute_word_accuracy(y_true, y_pred, tokens):
    # count to keep track of correct predictions
    count = 0
    for t, y in zip(y_true, y_pred):
        # s_t and s_y are the target and prediction
        s_y, s_t = '', ''
        for i in y:
            c = tokens[int(i)]
            # if we encounter stop-token, stop forming the word
            if c == '>':
                break 
            # else add the character to the string
            s_y += c
        s_y = s_y.strip()
        for i in t:
            c = tokens[int(i)]
            # if we encounter stop-token, stop forming the word
            if c == '>':
                break 
            # else add the character to the string
            s_t += c
        s_t = s_t.strip()
        # check if the target word == predicted word
        count += int(s_t == s_y)
    # to compute accuracy, divide by total number of items in the dataset
    return count/len(y_true)

# a function to read the data into a pd dataframe
def load_data(path):
    data = pd.read_csv(path, 
                       sep='\t',
                       encoding="utf8",
                       names=["hi","en","_"], 
                       skip_blank_lines=True)
                           
    data = data[data['hi'].notna()]
    data = data[data['en'].notna()]
    data = data[['hi','en']]
    return data

# a function to preprocess the data
def pre_process(data, max_eng_len, max_hin_len, eng_token_map, hin_token_map):
    x = data['en'].values 
    # add start and end tokens to the hindi word
    y = '<' + data['hi'].values + '>'
    
    # a is the encoder input
    a = np.zeros((len(x), max_eng_len))
    # b is the decoder input (has start-token and end-token)
    b = np.zeros((len(y), max_hin_len))
    # c is the decoder output, which leads the decoder input by one step
    # as it does not have start token in the beginning
    c = np.zeros((len(y), max_hin_len))
    
    # replace the characters by numbers so that the model can process them
    # use a inverted_index to map the characters to integers
    # these integers are just the index when the vocabulary characters are sorted
    for i, (xx, yy) in enumerate(zip(x, y)):
        for j, ch in enumerate(xx):
            a[i, j] = eng_token_map[ch]
        for j, ch in enumerate(yy):
            b[i, j] = hin_token_map[ch]
            if j > 0:
                # this is why c leads by one time-step
                c[i, j-1] = hin_token_map[ch]
    return a, b, c

##########################################################################################################################

# function to run the model with cmdline arguments
def build_and_run_model_with_cmd_args():
    # load the train and validation data
    train = load_data(train_dir)
    dev = load_data(dev_dir)

    # add start and end tokens to the hindi word
    # now generate the english and hindi vocabulary
    x = train['en'].values
    y = '<' + train['hi'].values + '>'

    # get the set of all unique characters, i.e. the vocabulary
    eng_tokens = set()
    hin_tokens = set()
    for xx, yy in zip(x,y):
        for ch in xx:
            eng_tokens.add(ch)
        for ch in yy:
            hin_tokens.add(ch)
    
    # sort the characters and create a inverted_index 
    # to map the characters to their index in the vocabulary
    eng_tokens = sorted(list(eng_tokens))
    hin_tokens = sorted(list(hin_tokens))
    eng_token_map = dict([(ch, i+1) for i, ch in enumerate(eng_tokens)])
    hin_token_map = dict([(ch, i+1) for i, ch in enumerate(hin_tokens)])
    # make sure to add the pad-token to the vocabulary as well
    # it should always be at the 0th index
    eng_tokens.insert(0, ' ')
    hin_tokens.insert(0, ' ')
    hin_token_map[' '] = 0
    eng_token_map[' '] = 0
    # compute the max english word and hindi word sizes
    # we will need this for decoder predictions (sequence length)
    max_eng_len = max([len(xx) for xx in x])
    max_hin_len = max([len(yy) for yy in y])

    # get the training encoder input, decoder input and decoder target
    trainxe, trainxd, trainy = pre_process(train, 
                                           max_eng_len, 
                                           max_hin_len, 
                                           eng_token_map, 
                                           hin_token_map)

    # get the validation encoder input, decoder input and decoder target
    valxe, valxd, valy = pre_process(dev, 
                                     max_eng_len, 
                                     max_hin_len, 
                                     eng_token_map, 
                                     hin_token_map)

    # create the encoder with the required hyperparameters
    encoder = Encoder(input_dim=int(trainxe.max())+1,
                      embed_dim=args.enc_embedding_size,
                      cell_hidden_dim=args.num_recurrent_units,
                      dropout=args.dropout,
                      k=args.num_enc_recurrent_layers, 
                      cell_type=args.recurrent_layer)

    # create the decoder with the required hyperparameters
    decoder = Decoder(input_dim=int(trainxd.max())+1, 
                      output_dim=int(trainy.max())+1, 
                      embed_dim=args.dec_embedding_size,
                      cell_hidden_dim=args.num_recurrent_units,
                      dropout=args.dropout,
                      k=args.num_dec_recurrent_layers, 
                      cell_type=args.recurrent_layer)

    # create the transliteration model with the created encoder and decoder
    model = TransliterationModel(encoder=encoder, 
                                 decoder=decoder, 
                                 tgt_max_len=max_hin_len)

    # instantiate and use the required optimizer
    optimizer = {
        "ranger": Lookahead(RectifiedAdam(learning_rate=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)),
        "adamw": AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True),
        "sgdw": SGDW(learning_rate=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9, nesterov=True),
        "novograd": NovoGrad(learning_rate=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
    }[args.optimizer]

    # define early stopping to terminate the run if the validation accuracy drops
    # continously for 4 times
    early_stop = EarlyStopping(monitor="val_accuracy",
                               patience=4, 
                               min_delta=1e-3, 
                               restore_best_weights=True)

    # compile the model and fit it to the data
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])

    model.fit([trainxe, trainxd], 
              trainy, 
              epochs=25, 
              callbacks=[early_stop],
              batch_size=args.batch_size, 
              validation_data=([valxe, valxd], valy), 
              shuffle=True)
    
    # after the run is completed, check the train word-level accuracy 
    # for prediction, pass the encoder input to the encoder and start-token to the decoder
    pred = model.predict([trainxe, trainxd[:, 0]], batch_size=args.batch_size)
    acc = compute_word_accuracy(pred.tolist(), trainy.tolist(), hin_tokens)
    print(f"train_word_level_accuracy: {acc:.4f}")

    # check validation-level accuracy as well
    # for prediction, pass the encoder input to the encoder and start-token to the decoder
    pred = model.predict([valxe, valxd[:, 0]], batch_size=args.batch_size)
    acc = compute_word_accuracy(pred.tolist(), valy.tolist(), hin_tokens)
    print(f"validation_word_level_accuracy: {acc:.4f}")

########################################################################################################################

# function to run the model with wandb 
def build_and_run_model_with_wandb(config=None):
    run = wandb.init(config=config)
    config = wandb.config

    # rename the run
    wandb.run.name = (
        f"layer_{config.recurrent_layer}"
        f"_num-recurrent-units_{config.num_recurrent_units}"
        f"_enc-embed-dim_{config.enc_embed_dim}"
        f"_dec-embed-dim_{config.dec_embed_dim}"
        f"_dropout_{config.dropout}"
        f"_num-enc-recurrent-layers_{config.num_enc_recurrent_layers}"
        f"_num-dec-recurrent-layers_{config.num_dec_recurrent_layers}"
        f"_optimizer_{config.optimizer}"
        f"_lr_{config.lr}"
        f"_weight_decay_{config.weight_decay}"
        f"_batch-size_{config.batch_size}"
    )

    # load the train and validation data
    train = load_data(train_dir)
    dev = load_data(dev_dir)

    # add start and end tokens to the hindi word
    # now generate the english and hindi vocabulary
    x = train['en'].values
    y = '<' + train['hi'].values + '>'

    # get the set of all unique characters, i.e. the vocabulary
    eng_tokens = set()
    hin_tokens = set()
    for xx, yy in zip(x,y):
        for ch in xx:
            eng_tokens.add(ch)
        for ch in yy:
            hin_tokens.add(ch)
        
    # sort the characters and create a inverted_index 
    # to map the characters to their index in the vocabulary
    eng_tokens = sorted(list(eng_tokens))
    hin_tokens = sorted(list(hin_tokens))
    eng_token_map = dict([(ch, i+1) for i, ch in enumerate(eng_tokens)])
    hin_token_map = dict([(ch, i+1) for i, ch in enumerate(hin_tokens)])
    eng_tokens.insert(0, ' ')
    hin_tokens.insert(0, ' ')
    eng_token_map[' '] = 0
    hin_token_map[' '] = 0
    max_eng_len = max([len(xx) for xx in x])
    max_hin_len = max([len(yy) for yy in y])

    # get the training encoder input, decoder input and decoder target
    trainxe, trainxd, trainy = pre_process(train, 
                                           max_eng_len, 
                                           max_hin_len, 
                                           eng_token_map, 
                                           hin_token_map)

    # get the validation encoder input, decoder input and decoder target
    valxe, valxd, valy = pre_process(dev, 
                                     max_eng_len, 
                                     max_hin_len, 
                                     eng_token_map, 
                                     hin_token_map)

    # create the encoder with the required wandb hyperparameters
    encoder = Encoder(input_dim=int(trainxe.max())+1,
                      embed_dim=config.enc_embed_dim,
                      cell_hidden_dim=config.num_recurrent_units,
                      dropout=config.dropout,
                      k=config.num_enc_recurrent_layers, 
                      cell_type=config.recurrent_layer)

    # create the decoder with the required wandb hyperparameters
    decoder = Decoder(input_dim=int(trainxd.max())+1, 
                      output_dim=int(trainy.max())+1, 
                      embed_dim=config.dec_embed_dim,
                      cell_hidden_dim=config.num_recurrent_units,
                      dropout=config.dropout,
                      k=config.num_dec_recurrent_layers, 
                      cell_type=config.recurrent_layer)

    # create the transliteration model with the created encoder and decoder
    model = TransliterationModel(encoder=encoder, 
                                 decoder=decoder, 
                                 tgt_max_len=max_hin_len)

    # instantiate and use the required optimizer
    optimizer = {
        "ranger": Lookahead(RectifiedAdam(learning_rate=config.lr, weight_decay=config.weight_decay, amsgrad=True)),
        "adamw": AdamW(learning_rate=config.lr, weight_decay=config.weight_decay, amsgrad=True),
        "sgdw": SGDW(learning_rate=config.lr, weight_decay=config.weight_decay, momentum=0.9, nesterov=True),
        "novograd": NovoGrad(learning_rate=config.lr, weight_decay=config.weight_decay, amsgrad=True)
    }[config.optimizer]

    # define early stopping to terminate the run if the validation accuracy drops
    # continously for 4 times
    early_stop = EarlyStopping(monitor="val_accuracy",
                               patience=4,
                               restore_best_weights=True,
                               min_delta=1e-3)

    # comile the model and fit it to the data
    model.compile(optimizer=optimizer, 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])

    model.fit([trainxe, trainxd], 
              trainy, 
              epochs=25, 
              callbacks=[early_stop, WandbCallback(save_model=False)],
              batch_size=config.batch_size, 
              validation_data=([valxe, valxd], valy), 
              shuffle=True)

    # after the run is completed, check the train word-level accuracy 
    # for prediction, pass the encoder input to the encoder and start-token to the decoder
    pred = model.predict([trainxe, trainxd[:, 0]], batch_size=config.batch_size)
    word_accuracy = compute_word_accuracy(pred.tolist(), trainy.tolist(), hin_tokens)

    # check validation-level accuracy as well
    # for prediction, pass the encoder input to the encoder and start-token to the decoder
    pred = model.predict([valxe, valxd[:, 0]], batch_size=config.batch_size)
    val_word_accuracy = compute_word_accuracy(pred.tolist(), valy.tolist(), hin_tokens)
    
    # log the word-level accuracy
    wandb.log({
        "word_accuracy": word_accuracy,
        "val_word_accuracy": val_word_accuracy
    })
              
    run.finish()

########################################################################################################################

# driver code
# run the required function according to wandb flag
if not args.wandb:
    build_and_run_model_with_cmd_args()
else:
    # setup the sweep config
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_word_accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'values': [1e-2, 1e-3, 1e-4]
            },
            'batch_size': {
                'values': [128, 256, 512]
            },
            'num_enc_recurrent_layers': {
                'values': [1, 2, 3]
            },
            'num_dec_recurrent_layers': {
                'values': [1, 2, 3]
            },
            'num_recurrent_units': {
                'values': [64, 128, 256],
            },
            'dropout': {
                'values': [0.1, 0.2, 0.3]
            },
            'recurrent_layer': {
                'values': ["rnn", "gru", "lstm"],
            },
            'weight_decay': {
                'values': [1e-4, 1e-3, 1e-2],
            },
            'enc_embed_dim': {
                'values': [64, 128, 256],
            },
            'dec_embed_dim': {
                'values': [64, 128, 256],
            },
            'optimizer': {
                'values': ["ranger", "adamw", "sgdw", "novograd"],
            },
        }
    }

    # call the sweep
    sweep_id = wandb.sweep(sweep_config, project='IITM-CS6910-Projects-Seq2Seq')
    wandb.agent(sweep_id, function=build_and_run_model_with_wandb)