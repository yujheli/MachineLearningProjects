
# coding: utf-8
import pandas as pd
import numpy  as np
import sys


train = pd.read_csv(sys.argv[1])


def splt(s):
    return [float(i)/255 for i in s.split(' ')]

data = list(train['feature'].apply(splt))
x_train = np.array(data)



np.save("mean_value", np.mean(x_train, axis=0))

x_train -= np.mean(x_train, axis=0)
np.save("mean_std", np.std(x_train, axis=0))

x_train /= np.std(x_train, axis=0)
x_train = x_train.reshape(x_train.shape[0],48,48,1)


x_train = x_train.reshape(x_train.shape[0],48,48,1)

train_label = np.array(train['label'])

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit(np.array(range(7)).reshape(7,1))


train_label = enc.transform(train_label.reshape(-1,1)).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_train, train_label, test_size=0.1)

from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

from keras.models import Sequential
from keras.layers import *
from keras import callbacks
# from keras.callbacks import TensorBoard
from keras.optimizers import SGD,Adam
# Generate dummy data

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu',padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu',padding ='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-4)
#sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
#callbacks = callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=4, verbose=0, mode='auto')
#tensorboard = callbacks.TensorBoard(log_dir='./logs',histogram_freq=1, write_graph=False,  write_images=False)
for i in range(100):
    #history = model.fit(X_train, y_train, batch_size=400, epochs=200, validation_data=(X_test, y_test))

    history = model.fit_generator(datagen.flow(X_train, y_train,
                    batch_size=400),
                    nb_epoch=200,
                    validation_data=(X_test, y_test),
                    samples_per_epoch=X_train.shape[0])


    score = model.evaluate(X_train, y_train)
    val_loss = model.evaluate(X_test, y_test)
    model_name = 'model_Jack_acc_{:8f}_{:8f}.model'.format(val_loss[1],score[1])
    model.save(model_name)



