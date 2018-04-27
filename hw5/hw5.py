import argparse
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import add, Dot, Input, Dense, Lambda, Reshape, Dropout, Embedding, Concatenate
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.externals import joblib


def parse_args():
    parser = argparse.ArgumentParser('Matrix Factorization.')
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--dnn', type=int, nargs='*')
    parser.add_argument('--norm', type=int)


    return parser.parse_args()




def read_data(trainfile, testfile):
    traindf, testdf = pd.read_csv(trainfile), pd.read_csv(testfile)

    traindf['test'] = 0
    testdf['test'] = 1

    df = pd.concat([traindf, testdf])

    id2user = df['UserID'].unique()
    id2movie = df['MovieID'].unique()

    user2id = {k: id for id, k in enumerate(id2user)}
    movie2id = {k: id for id, k in enumerate(id2movie)}

    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])

    df_train = df.loc[df['test'] == 0]

    return df_train[['UserID', 'MovieID']].values, df_train['Rating'].values, df[['UserID', 'MovieID']].values, user2id, movie2id


def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


class WeightedAvgOverTime(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedAvgOverTime, self).__init__(**kwargs)
   
    def call(self, x, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, axis=-1)
            s = K.sum(mask, axis=1)
            if K.equal(s, K.zeros_like(s)) is None:
                return K.mean(x, axis=1)
            else:
                return K.cast(K.sum(x * mask, axis=1) / K.sqrt(s), K.floatx())
        else:
            return K.mean(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def compute_mask(self, x, mask=None):
        return None

    def get_config(self):
        base_config = super(WeightedAvgOverTime, self).get_config()
        return dict(list(base_config.items()))


def build_MF(num_users, num_movies, dim, feedback_u, feedback_m):
    u_input = Input(shape=(1,))
    U = Embedding(num_users, dim, embeddings_regularizer=l2(0.00001))(u_input)
    U = Reshape((dim,))(U)
    U = Dropout(0.1)(U)

    m_input = Input(shape=(1,))
    M = Embedding(num_movies, dim, embeddings_regularizer=l2(0.00001))(m_input)
    M = Reshape((dim,))(M)
    M = Dropout(0.1)(M)

    
    # F_u = Reshape((feedback_u.shape[1],))(Embedding(num_users, feedback_u.shape[1], trainable=False, weights=[feedback_u])(u_input))
    # F_u = Embedding(num_movies+1, dim, embeddings_initializer=Zeros(), embeddings_regularizer=l2(0.00001), mask_zero=True)(F_u)
    # F_u = Dropout(0.1)(F_u)
    # F_u = WeightedAvgOverTime()(F_u)

    # U = add([U, F_u])
    
    # F_m = Reshape((feedback_m.shape[1],))(Embedding(num_movies, feedback_m.shape[1], trainable=False, weights=[feedback_m])(m_input))
    # F_m = Embedding(num_users+1, dim, embeddings_initializer=Zeros(), embeddings_regularizer=l2(0.00001), mask_zero=True)(F_m)
    # F_m = Dropout(0.1)(F_m)
    # F_m = WeightedAvgOverTime()(F_m)

    # M = add([M, F_m])
    
    pred = Dot(axes=-1)([U, M])
    U_bias = Reshape((1,))(Embedding(num_users, 1, embeddings_regularizer=l2(0.00001))(u_input))
    M_bias = Reshape((1,))(Embedding(num_users, 1, embeddings_regularizer=l2(0.00001))(m_input))

    pred = add([pred, U_bias, M_bias])
    pred = Lambda(lambda x: x + K.constant(3.5817, dtype=K.floatx()))(pred)
        
    return Model(inputs=[u_input, m_input], outputs=[pred])



def build_DNN(num_users, num_movies, dim, feedback_u, feedback_m, dnn):
    u_input = Input(shape=(1,))
    U = Embedding(num_users, dim, embeddings_regularizer=l2(0.00001))(u_input)
    U = Reshape((dim,))(U)
    U = Dropout(0.1)(U)

    m_input = Input(shape=(1,))
    M = Embedding(num_movies, dim, embeddings_regularizer=l2(0.00001))(m_input)
    M = Reshape((dim,))(M)
    M = Dropout(0.1)(M)

    
    pred = Concatenate()([U, M])
    for units in dnn:
        pred = Dense(units, activation='relu')(pred)
        pred = Dropout(0.3)(pred)

    pred = Dense(1, activation='relu')(pred)
        
    return Model(inputs=[u_input, m_input], outputs=[pred])


def get_feedback(X, num_users, num_movies):
    feedback_u = [[] for u in range(num_users)]
    feedback_m = [[] for i in range(num_movies)]

    for u, m in zip(X[:, 0], X[:, 1]):
        feedback_u[u].append(m+1)
        feedback_m[m].append(u+1)

    return feedback_u, feedback_m

args = parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
K.tensorflow_backend.set_session(tf.Session(config=config))

X_train, Y_train, X, user2id, movie2id = read_data(args.train, args.test)

norm_flag = 0
if args.norm is not None:
    Y_train = Y_train-np.mean(Y_train)
    Y_train = Y_train/np.std(Y_train)
    print("Doing normalizing")
    norm_flag = 1


num_users, num_movies = len(user2id), len(movie2id)

feedback_u, feedback_m = get_feedback(X, num_users, num_movies)
feedback_u, feedback_m = pad_sequences(feedback_u), pad_sequences(feedback_m)

np.save('user2id', user2id)
np.save('movie2id', movie2id)

np.random.seed(5)
indices = np.random.permutation(len(X_train))
X_train, Y_train = X_train[indices], Y_train[indices]

dim = args.dim



callbacks = []
callbacks.append(EarlyStopping(monitor='val_rmse', patience=100))


if args.dnn is not None:
    model = build_DNN(num_users, num_movies, dim, feedback_u, feedback_m, args.dnn)
    callbacks.append(ModelCheckpoint('model_dnn_'+str(args.dim)+'_'+str(norm_flag)+'.h5', monitor='val_rmse', save_best_only=True))
    print("Building DNN")
else:
    model = build_MF(num_users, num_movies, dim, feedback_u, feedback_m)
    callbacks.append(ModelCheckpoint('model_MF_'+str(args.dim)+'_'+str(norm_flag)+'.h5', monitor='val_rmse', save_best_only=True))
    print("Building MF")

model.summary()



model.compile(loss='mse', optimizer='adam', metrics=[rmse])
his = model.fit([X_train[:, 0], X_train[:, 1]], Y_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=callbacks) 

# joblib.dump(t,'pun_tokenizer.pkl')
# joblib.dump(his,'his_dim_'+str(args.dim)+'.pkl')

if args.dnn is not None:
    joblib.dump(his.history,'his_dnn_'+str(args.dim)+'_'+str(norm_flag)+'.pkl')
else:
    joblib.dump(his.history,'his_MF_'+str(args.dim)+'_'+str(norm_flag)+'.pkl')

    


