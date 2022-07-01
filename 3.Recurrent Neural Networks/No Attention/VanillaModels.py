# import statements
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

# define encoder using sub-classing
class Encoder(Model):
    
    def __init__(self, input_dim, embed_dim=128, k=1, cell_hidden_dim=256, dropout=0.25, cell_type="rnn"):
        super(Encoder, self).__init__()
        # recurrent layers as a list
        self.cells = []
        # type of cell (rnn/gru/lstm)
        self.cell_type = cell_type
        # embedding layer
        # set mask_zero as True to avoid model focusing only on padding tokens when they are in majority
        self.embedding = Embedding(input_dim=input_dim, output_dim=embed_dim, mask_zero=True)

        # add the required number of recurrent layers to the encoder
        # always make sure return_state is set to true
        for _ in range(k):
            if cell_type == "rnn":
                self.cells.append(SimpleRNN(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
            elif cell_type == "gru":
                self.cells.append(GRU(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
            elif cell_type == "lstm":
                self.cells.append(LSTM(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
    
    # define custom call method for encoder
    def call(self, x):
        # first pass input through embedding
        x = self.embedding(x)
        # pass the output through each recurrent layer
        for cell in self.cells:
            if self.cell_type == "lstm":
                x, state_h, state_c = cell(x)
                state = [state_h, state_c]
            else:
                # gru/rnn have a single state
                # while lstm has two states
                x, state = cell(x)
        # return both state and output for the encoder
        # this states needs to be passed to the decoder
        return x, state

##########################################################################################################################

#define decoder using sub-classing
class Decoder(Model):
    
    def __init__(self, input_dim, output_dim, embed_dim=128, k=1, cell_hidden_dim=256, dropout=0.25, cell_type="rnn"):
        super(Decoder, self).__init__()
        # recurrent layers as a list
        self.cells = []
        # type of cell (rnn/gru/lstm)
        self.cell_type = cell_type
        # final fc layer with softmax to predict distribution over the vocabulary
        self.fc = Dense(output_dim, activation="softmax")
        # embedding layer
        # set mask_zero as True to avoid model focusing only on padding tokens when they are in majority
        self.embedding = Embedding(input_dim=input_dim, output_dim=embed_dim, mask_zero=True)
        
        # add the required number of recurrent layers to the decoder
        # always make sure return_state and return_sequences is set to true
        for _ in range(k):
            if cell_type == "rnn":
                self.cells.append(SimpleRNN(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
            elif cell_type == "gru":
                self.cells.append(GRU(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
            elif cell_type == "lstm":
                self.cells.append(LSTM(cell_hidden_dim, return_sequences=True, return_state=True, dropout=dropout))
    
    # define custom call method for decoder
    def call(self, x, prev_state):
        # first pass input through embedding
        x = self.embedding(x)
        # pass the output through each recurrent layer
        # with previous state
        for cell in self.cells:
            if self.cell_type == "lstm":
                x, state_h, state_c = cell(x, initial_state=prev_state)
                state = [state_h, state_c]
            else:
                # gru/rnn have a single state
                # while lstm has two states
                x, state = cell(x, initial_state=prev_state)
        # pass the final decoder output to the fc layer
        # to get a distribution over the vocabulary
        x = self.fc(x)
        # return the output and decoder state
        # this state will be passed to next decoder timestep
        return x, state

########################################################################################################################

# define final transliteration model using subclassing
class TransliterationModel(Model):
    
    # a transliteration model has both encoder and decoder
    def __init__(self, encoder, decoder, tgt_max_len):
        super(TransliterationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_max_len = tgt_max_len

    # define custom call method to help with `predict` function
    def call(self, x):
        # input consists of encoder input (english word)
        # and the <START> token (which is `<`) in this case
        # function outputs complete transliterated word
        (xe, pred_t) = x
        preds = []
        # first, get the state from the encoder
        # set training to false, to run it in inference mode
        _, state = self.encoder(xe, training=False)
        # sequentially run the decoder and predict the next tokens one-by-one
        for _ in range(self.tgt_max_len):
            # make sure the input is in the correct dimension
            pred_t = tf.expand_dims(pred_t, axis=1)
            # get the current timestep decoder output
            pred_t, state = self.decoder(pred_t, state, training=False)
            pred_t = tf.squeeze(pred_t, axis=1)
            # from the distribution predicted, take the most probable token
            pred_t = tf.argmax(pred_t, axis=1)
            # append it to the predictions list to get the full hindi word at the end
            # as input from next timestep, i.e. no teacher forcing for prediction
            preds.append(pred_t)
        preds = tf.convert_to_tensor(preds)
        preds = tf.transpose(preds, perm=[1, 0])
        return preds

    # override `test_step` to perform custom validation step
    def test_step(self, data):
        # use the first decoder input which is start-tokens
        # use decoder target to compute the metrics
        ((xe, xd), y) = data
        preds, pred_t, loss = [], xd[:, 0], 0 
        # get encoder state after parsing english word
        # make sure to set training as False to run the model in inference mode
        _, state = self.encoder(xe, training=False)
        # sequentially run the decoder and predict the next tokens one-by-one
        for t in range(self.tgt_max_len):
            # make sure the input and target are in the correct dimension
            pred_t = tf.expand_dims(pred_t, axis=1)
            y_t = tf.expand_dims(y[:, t], axis=1)
            # get the current timestep decoder output
            pred_t, state = self.decoder(pred_t, state, training=False)
            pred_t = tf.squeeze(pred_t, axis=1)
            # compute the validation loss using the prediction and target
            loss += self.compiled_loss(y_t, pred_t)
            # append it to the predictions list to get the full hindi word at the end
            preds.append(pred_t)
            # from the distribution predicted, take the most probable token 
            # as input from next timestep, i.e. no teacher forcing for validation step
            pred_t = tf.argmax(pred_t, axis=1)
        preds = tf.convert_to_tensor(preds)
        preds = tf.transpose(preds, perm=(1, 0, 2))
        # update the compiled_metrics (accuracy/loss) once the step is completed
        self.compiled_metrics.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

    # override `train_step` to perform custom training
    def train_step(self, data):
        # y is always one timestep ahead of xd
        # xd is decoder input, and y is decoder target
        ((xe, xd), y) = data
        preds, loss = [], 0
        # initialize Gradient Tape to keep track of operations and gradients
        # all learning must happen inside this scope
        with tf.GradientTape() as tape:
            # get encoder state after parsing english word
            _, state = self.encoder(xe, training=True)
            # sequentially run the decoder and predict the next tokens one-by-one
            for t in range(self.tgt_max_len):
                # make sure the input and target are in the correct dimension
                # here the input to the decoder is the correct previous timestep prediction
                # i.e. teacher-forcing is being used
                xd_t = tf.expand_dims(xd[:, t], axis=1)
                y_t = tf.expand_dims(y[:, t], axis=1)
                # get the current timestep decoder output
                pred_t, state = self.decoder(xd_t, state, training=True)
                pred_t = tf.squeeze(pred_t, axis=1)
                # compute the training loss using the prediction and target
                loss += self.compiled_loss(y_t, pred_t)
                # append it to the predictions list to get the full hindi word at the end
                preds.append(pred_t)

        # use the encoder and decoder variables to compute the gradients
        vars = self.encoder.trainable_variables + \
               self.decoder.trainable_variables
        # compute the gradients from the operations recorded by Tape
        # backprop step
        grads = tape.gradient(loss, vars)
        # apply the gradients to the encoder and decoder
        self.optimizer.apply_gradients(zip(grads, vars))
        preds = tf.convert_to_tensor(preds)
        preds = tf.transpose(preds, perm=(1, 0, 2))
        # update the compiled_metrics (accuracy/loss) once the step is completed
        self.compiled_metrics.update_state(y, preds)
        return {m.name: m.result() for m in self.metrics}

###########################################################################################################################