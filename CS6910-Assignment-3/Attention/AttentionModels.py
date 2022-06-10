# import statements
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

# define attention using sub-classing
class Attention(Layer):
    def __init__(self, attn_dim=256):
        super(Attention, self).__init__()
        self.V = Dense(1, use_bias=False)
        self.U = Dense(attn_dim, use_bias=False)
        self.W = Dense(attn_dim, use_bias=False)

    def call(self, h_j, s_t):
        f1 = self.U(h_j)
        f2 = tf.expand_dims(self.W(s_t), axis=1)
        # e_t = v^T tanh(W*h_i + U*s_t)
        e_t = self.V(tf.nn.tanh(f1 + f2))
        # a_t = softmax(e_t)
        a_t = tf.nn.softmax(e_t, axis=1)
        # c_t = \sigma_t a_t * h_i
        c_t = tf.reduce_sum(a_t * h_j, axis=1)
        c_t =  tf.expand_dims(c_t, axis=1)
        # return both context vector and attention weights
        return c_t, a_t

########################################################################################################################

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

########################################################################################################################

#define decoder using sub-classing
class AttentionDecoder(Model):
    def __init__(self, input_dim, output_dim, embed_dim=128, attn_dim=256, k=1, cell_hidden_dim=256, dropout=0.25, cell_type="lstm"):
        super(AttentionDecoder, self).__init__()
        # recurrent layers as a list
        self.cells = []
        # type of cell (rnn/gru/lstm)
        self.cell_type = cell_type
        # final fc layer with softmax to predict distribution over the vocabulary
        self.fc = Dense(output_dim, activation="softmax")
        # bahdanau attention layer
        self.attention = Attention(attn_dim=attn_dim)
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
    def call(self, x, h_j, state):
        # first pass input through embedding
        x = self.embedding(x)
        for i, cell in enumerate(self.cells):
            # use the final nth recurrent layer to get
            # the context vector and attention
            # using the abstract representation from the previous n-1 layers
            if i == len(self.cells) - 1:
                if self.cell_type == "lstm":
                    # use only state_h for attention
                    # hence pass state[0] which is state_h
                    c_t, a_t = self.attention(h_j, state[0])
                else:
                    # rnn/gru have only one state
                    # directly use that for attention
                    c_t, a_t = self.attention(h_j, state)
                x = tf.concat([x, c_t], axis=-1)
            
            # use the n-1 initial recurrent layers to create 
            # more abstract sequential representations from the input
            if self.cell_type == "lstm":
                x, state_h, state_c = cell(x, initial_state=state)
                state = [state_h, state_c]
            else:
                # gru/rnn have a single state
                # while lstm has two states
                x, state = cell(x, initial_state=state)
        # pass the final decoder output to the fc layer
        # to get a distribution over the vocabulary
        x = self.fc(x)
        # return the output and decoder state
        # this state will be passed to next decoder timestep
        return x, state, a_t

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
        # first, get the state from the encoder
        # set training to false, to run it in inference mode
        preds, attn_weights = [], []
        h_i, state = self.encoder(xe, training=False)
        # sequentially run the decoder and predict the next tokens one-by-one
        for _ in range(self.tgt_max_len):
            # make sure the input is in the correct dimension
            pred_t = tf.expand_dims(pred_t, axis=1)
            # get the current timestep decoder output
            # along with attention weights for that timestep
            pred_t, state, a_t = self.decoder(pred_t, h_i, state, training=False)
            pred_t = tf.squeeze(pred_t, axis=1)
            # from the distribution predicted, take the most probable token
            pred_t = tf.argmax(pred_t, axis=1)
            # append it to the predictions list to get the full hindi word at the end
            # as input from next timestep, i.e. no teacher forcing for prediction
            preds.append(pred_t)
            # similarly, keep a track of all the attention weights
            attn_weights.append(a_t)
        attn_weights = tf.convert_to_tensor(attn_weights)
        attn_weights = tf.squeeze(attn_weights, axis=-1)
        attn_weights = tf.transpose(attn_weights, perm=[1, 0, 2])
        preds = tf.convert_to_tensor(preds)
        preds = tf.transpose(preds, perm=[1, 0])
        return preds, attn_weights

    # override `test_step` to perform custom validation step
    def test_step(self, data):
        # use the first decoder input which is start-tokens
        # use decoder target to compute the metrics
        ((xe, xd), y) = data
        preds, pred_t, loss = [], xd[:, 0], 0 
        # get encoder state after parsing english word
        # make sure to set training as False to run the model in inference mode
        h_i, state = self.encoder(xe, training=False)
        # sequentially run the decoder and predict the next tokens one-by-one
        for t in range(self.tgt_max_len):
            # make sure the input and target are in the correct dimension
            pred_t = tf.expand_dims(pred_t, axis=1)
            y_t = tf.expand_dims(y[:, t], axis=1)
            # get the current timestep decoder output
            # in this step, attention weights are not required
            pred_t, state, _ = self.decoder(pred_t, h_i, state, training=False)
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
            h_i, state = self.encoder(xe, training=True)
            # sequentially run the decoder and predict the next tokens one-by-on
            for t in range(self.tgt_max_len):
                # make sure the input and target are in the correct dimension
                # here the input to the decoder is the correct previous timestep prediction
                # i.e. teacher-forcing is being used
                xd_t = tf.expand_dims(xd[:, t], axis=1)
                y_t = tf.expand_dims(y[:, t], axis=1)
                # get the current timestep decoder output
                # in this step we don't need the attention weights
                # so we can ignore them
                pred_t, state, _ = self.decoder(xd_t, h_i, state, training=True)
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

########################################################################################################################