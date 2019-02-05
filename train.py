import numpy as np
import pickle, sys, os

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import Adam
from crf import CRFLayer, create_custom_objects
from keras_contrib.layers import CRF
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import backend as K
from keras_contrib.layers import CRF
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# PARAMETERS ================
MAX_SEQUENCE_LENGTH = 55 #Calculated for this dataset
EMBEDDING_DIM = 300 #Fasttext hindi embeddigns
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
MODEL_TYPE = 2
#MODEL TYPE 1 = Bi-Lstm
#MODEL TYPE 2 = Bi-Lstm + CRF
dropout = 0.5
epochs = 100

with open('PickledData/data.pkl', 'rb') as f:
    X, y, word2int, int2word, tag2int, int2tag = pickle.load(f)


def generator(all_X, all_y, n_classes, batch_size=BATCH_SIZE):
    num_samples = len(all_X)

    while True:

        for offset in range(0, num_samples, batch_size):
            
            X = all_X[offset:offset+batch_size]
            #S = np.asarray([MAX_SEQUENCE_LENGTH] * batch_size, dtype='int32')
            y = all_y[offset:offset+batch_size]

            y = to_categorical(y, num_classes=n_classes)


            yield shuffle(X, y)

def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

n_tags = len(tag2int)

X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)

print('TOTAL TAGS', len(tag2int))
print('TOTAL WORDS', len(word2int))

# shuffle the data
X, y = shuffle(X, y)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT,random_state=42)

# split training data into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('We have %d TRAINING samples' % n_train_samples)
print('We have %d VALIDATION samples' % n_val_samples)
print('We have %d TEST samples' % n_test_samples)

# make generators for training and validation
train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_tags + 1)
validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_tags + 1)


with open('PickledData/Glove.pkl', 'rb') as f:
	embeddings_index = pickle.load(f)

print('Total %s word vectors.' % len(embeddings_index))

# + 1 to include the unkown word
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embeddings_index will remain unchanged and thus will be random.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

# Bi-Lstm 
if(MODEL_TYPE == 1):
  l_lstm = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.1))(embedded_sequences)
  preds = TimeDistributed(Dense(n_tags + 1, activation='softmax'))(l_lstm)
  model = Model(sequence_input, preds)
  model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy', ignore_class_accuracy(0)])
  print("model fitting - Bidirectional LSTM")

# Bi-Lstm + CRF
if(MODEL_TYPE == 2):
  l_lstm = Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.2))(embedded_sequences)
  model = TimeDistributed(Dense(50, activation='relu'))(l_lstm)
  crf = CRF(n_tags+1)  # CRF layer
  preds = crf(model)  # output
  model = Model(sequence_input, preds)

  model.compile(loss=crf.loss_function,
                optimizer=Adam(0.001),
                metrics=[crf.accuracy])
  print("model fitting - Bidirectional LSTM +CRF")


model.summary()
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

history = model.fit_generator(train_generator, 
                     steps_per_epoch=n_train_samples//BATCH_SIZE,
                     validation_data=validation_generator,
                     validation_steps=n_val_samples//BATCH_SIZE,
                     epochs=epochs,
                     callbacks=callbacks,
                     verbose=1,
                     workers=4)

hist = pd.DataFrame(history.history)

plt.style.use("ggplot")
plt.figure(figsize=(12,12))
plt.plot(hist["crf_viterbi_accuracy"], label="acc")
plt.plot(hist["val_crf_viterbi_accuracy"], label="val_acc")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
 fancybox=True, shadow=True)
plt.savefig('Bi-Lstm+CRF')
plt.show()


if not os.path.exists('Models/'):
    print('MAKING DIRECTORY Models/ to save model file')
    os.makedirs('Models/')

train = True

if train:
    model.save('Models/model.h5')
    print('MODEL SAVED in Models/ as model.h5')
else:
    from keras.models import load_model
    model = load_model('Models/model.h5')

y_test = to_categorical(y_test, num_classes=n_tags+1)
test_results = model.evaluate(X_test, y_test, verbose=0)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))

