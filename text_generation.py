import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM

text = open('data/data_clean.txt', 'r').read().lower()
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)
SEQ_LEN = 64
BATCH_SIZE = 16
BATCH_CHARS = len(text) / BATCH_SIZE
LSTM_SIZE = 256
LAYERS = 3


def build_model(infer):
    if infer:
        batch_size = seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(LSTM(LSTM_SIZE,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   consume_less='cpu',
                   unroll=True,
                   stateful=True))

    #model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        #model.add(Dropout(0.2))

    model.add(TimeDistributedDense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='nadam')
    return model


model = build_model(infer=True)
model.reset_states()
#model.load_weights('karpathy_keras_char_rnn.h5')
#model.load_weights('/tmp/keras_char_rnn.%d.h5' % epoch)t
