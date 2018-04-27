import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import SGD,Adam
import sys
batch_size = 512

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
with open('training_label.txt','rt') as inputfile:
    for line in inputfile.read().split('\n'):
        raw_data.append(line.split(' +++$+++ '))

raw_data = raw_data[:-1]
data = pd.DataFrame(raw_data)

word_size = 20000
t = Tokenizer(num_words=word_size)
#t = Tokenizer()

t.fit_on_texts(data.iloc[:,1])
vocab_size = word_size + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(data.iloc[:,1])


max_length = 36
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
train_label = np.array(data.iloc[:,0].apply(lambda x : float(x)))

X_train, Y_train, X_valid, Y_valid = split_valid_set(padded_docs, train_label, percentage=0.8)


print(len(X_train))
print(X_train[0])
print(len(Y_train))
print(Y_train[0])

print(len(X_valid))
print(X_valid[0])
print(len(Y_valid))
print(Y_valid[0])



model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
# model.add(Dropout(0.25))
model.add(Conv1D(64,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=2))
#model.add(Bidirectional(LSTM(128, return_sequences=True)))
#model.add(Bidirectional(LSTM(128)))
# model.add(LSTM(32,return_sequences=True))
model.add(LSTM(64))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='relu'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
#sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Train...')

for i in range(1):

    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=10,
              validation_data=[X_valid, Y_valid])

    # score = model.evaluate(X_train, Y_train)
    # val_loss = model.evaluate(X_valid, Y_valid)
    # model_name = 'model_acc_{:8f}_{:8f}.model'.format(val_loss[1],score[1])
    # model.save(model_name)




# print("loading the model......")
# model = load_model(sys.argv[1])


print("loading the unlabel data......")

raw_no_label = []
with open('training_nolabel.txt','rt') as inputfile:
    for line in inputfile.read().split('\n'):
        raw_no_label.append(line)

raw_no_label = raw_no_label[:-1]
data_no_label = pd.DataFrame(raw_no_label[:1000000])

print("padding the unlabel data......")


encoded_nolabel = t.texts_to_sequences(data_no_label.iloc[:,0])
padded_nolabel = pad_sequences(encoded_nolabel, maxlen=max_length, padding='post')

print("predicting the unlabel data......")

pre_label =model.predict(padded_nolabel,verbose=1)

threshold = 0.8
X_train = list(X_train)
Y_train = list(Y_train)

print("appending the unlabel data......")

for i in range(len(pre_label)):
    if pre_label[i]>threshold:
        X_train.append(list(padded_nolabel[i]))
        Y_train.append(1.0)
    elif pre_label[i]<(1-threshold):
        X_train.append(list(padded_nolabel[i]))
        Y_train.append(0.0)



X_train = np.array(X_train)
Y_train = np.array(Y_train)




model2 = Sequential()
model2.add(Embedding(vocab_size, 300, input_length=max_length))
# model.add(Dropout(0.25))
model2.add(Conv1D(64,
                 3,
                 padding='valid',
                 activation='relu',
                 strides=1))
model2.add(MaxPooling1D(pool_size=2))
#model.add(Bidirectional(LSTM(128, return_sequences=True)))
#model.add(Bidirectional(LSTM(128)))
# model.add(LSTM(32,return_sequences=True))
model2.add(LSTM(64))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
model2.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model2.add(Dense(1, activation='relu'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
#sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




print("Starting to train......")


for i in range(2):

    model2.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=6,
              validation_data=[X_valid, Y_valid])

    score = model2.evaluate(X_train, Y_train)
    val_loss = model2.evaluate(X_valid, Y_valid)
    model_name = 'model_semi_{:8f}_{:8f}.model'.format(val_loss[1],score[1])
    model2.save(model_name)



