import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import SGD,Adam
from sklearn.externals import joblib
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
batch_size = 128
import sys

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(np.floor(all_data_size * percentage))

    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid



raw_data = []
counter = 0
with open(sys.argv[1],'rt') as inputfile:
    for line in inputfile.read().split('\n'):
        raw_data.append(line.split(' +++$+++ '))

raw_data = raw_data[:-1]
data = pd.DataFrame(raw_data)

word_size = 20000
vocab_size = word_size + 1


# t = Tokenizer(num_words=word_size, filters='\t\n')
#t = Tokenizer()

# test_data = []
# count = 0
# with open('testing_data.txt','rt') as inputfile:
#     for line in inputfile.read().split('\n'):
#         if count==0:
#             count+=1
#         else:
#             xx = []
#             tmp = line.split(',')
#             xx.append(tmp[0])
#             xx.append(','.join(tmp[1:]))
#             test_data.append(xx)
            
# test_data = test_data[:-1]   
# test = pd.DataFrame(test_data)

# test_text = list(test.iloc[:,1])
train_text = list(data.iloc[:,1])

# from gensim.models import Word2Vec
# embed = Word2Vec([text_to_word_sequence(i,filters='\t\n') for i in test_text+train_text],  size=128, workers = 4, window=10, min_count=1)

# t.fit_on_texts(test_text+train_text)
# joblib.dump(t,'pun_tokenizer.pkl')


# embedding = np.zeros((vocab_size, 128))
# for word, index in t.word_index.items():
#     if index>word_size:
#         break
#     # elif word not in embed:
#     #     embedding[index] = embed[0]
#     else:
#         embedding[index] = embed[word]


# np.save('embed_pun',embedding)

# print("saved")

embedding = np.load('embed_pun.npy')
t = joblib.load('pun_tokenizer.pkl')
# t.fit_on_texts(data.iloc[:,1])



# integer encode the documents
encoded_docs = t.texts_to_sequences(data.iloc[:,1])


max_length = 36
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
train_label = np.array(data.iloc[:,0].apply(lambda x : float(x)))

X_train, Y_train, X_valid, Y_valid = split_valid_set(padded_docs, train_label, percentage=0.95)


print(len(X_train))
print(X_train[0])
print(len(Y_train))
print(Y_train[0])

print(len(X_valid))
print(X_valid[0])
print(len(Y_valid))
print(Y_valid[0])


model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_length,weights=[embedding],trainable=False))
# model.add(Dropout(0.25))
# model.add(Conv1D(64,
#                  3,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# # model.add(MaxPooling1D(pool_size=2))
# model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
# model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
# model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.4)))
# model.add(Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
model.add(Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
# model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.4)))
model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.4)))

# model.add(LSTM(512,return_sequences=True))
# model.add(LSTM(256,return_sequences=True))
# model.add(LSTM(128))
# model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
#sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
callbacks = [] 
# callbacks.append(EarlyStopping(monitor='val_fmeasure', patience=25, verbose=1, mode='max'))
callbacks.append(ModelCheckpoint('model-pun-gensim.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

print('Train...')

# for i in range(200):
model.summary()
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=200,
          validation_data=[X_valid, Y_valid], callbacks=callbacks)

score = model.evaluate(X_train, Y_train)
val_loss = model.evaluate(X_valid, Y_valid)

# model_name = 'model_acc_{:8f}_{:8f}.model'.format(val_loss[1],score[1])
# model.save(model_name)






