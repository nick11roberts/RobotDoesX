from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.layers import merge, Input
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, RMSprop
from keras.utils.data_utils import get_file
from datetime import datetime
import numpy as np
import rethinkdb as r
import random
import sys

# Connect to RethinkDB instance
r.connect("localhost", 28015, db="robot_does_x").repl()

# Path to latest weights
file_path = "model_latest.hdf5"

# Create instance id using current time
instance_id = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
instance_id = 'reslstm_nadam_20_' + instance_id
print("instance_id: ", instance_id)

# Open dataset
text = open('data/zoella.txt').read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        y[i, t, char_indices[next_chars[i]]] = 1


print('Build model...')
'''
model=Graph()
model.add_input(input_shape=(maxlen, len(chars)), name='input')
model.add_node(LSTM(128, return_sequences=True, init= 'orthogonal'), input='input', name='lstm1')
model.add_node(LSTM(128, return_sequences=True, init= 'orthogonal'), input='lstm1', name='lstm2')
model.add_node(TimeDistributed(Dense(len(chars), init= 'orthogonal')), input='lstm2', name='fc1')
model.add_node(TimeDistributed(Dense(len(chars), activation='softmax', init= 'orthogonal')), inputs=['fc1','input'], merge_mode='sum', name='fc2')
model.add_output(input='fc2', name='output')
'''
input_1 = Input(shape=(maxlen, len(chars)))
lstm_1 = LSTM(128, return_sequences=True, unroll=True, consume_less='cpu', init= 'orthogonal')(input_1)
#lstm_2 = LSTM(128, return_sequences=True, unroll=True, consume_less='cpu', init= 'orthogonal')(lstm_1)
timedistributed_1 = TimeDistributed(Dense(len(chars), init= 'orthogonal'))(lstm_1)
merge_1 = merge([input_1, timedistributed_1], mode='sum')
timedistributed_2 = TimeDistributed(Dense(len(chars), activation='softmax', init='orthogonal'))(merge_1)
model = Model(input=input_1, output=timedistributed_2)

optimizer = Nadam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Print model summary
print(model.summary())


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(1, 21):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, nb_epoch=1)

    # Save model to filesystem
    model.save_weights(file_path)
    # Open file to be stored in database
    fh = open(file_path, 'rb')
    contents = fh.read()
    fh.close()
    # Save model to RethinkDB
    r.table("train_output").insert([
        { "instance_id": instance_id,
        "model": r.binary(contents)
        }
    ]).run()

    start_index = random.randint(0, len(text) - maxlen - 1)

    if iteration % 1 == 0:
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0][-1]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

            # Save generated text to RethinkDB
            '''
            r.table("train_output").insert([
                { "instance_id": instance_id,
                "iteration": iteration, "diversity": diversity,
                "generated": generated
                }
            ]).run()
            '''
