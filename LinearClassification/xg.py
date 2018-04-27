
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from sklearn.externals import joblib

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(all_data_size * percentage)

    #X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid


def write_file(out_list, filename):
    with open(filename,'w') as output:
        output.write('id,label')
        counter = 1
        for i in out_list:
            output.write('\n')
            output.write(str(counter))
            output.write(',')
            output.write(str(int(i)))
            counter+=1


x = np.array(pd.read_csv(sys.argv[1]).values)
y = np.array(pd.read_csv(sys.argv[2]).values)

# X_test = np.array(pd.read_csv('X_test').values)
idx = np.random.permutation(len(x))
x,y = np.array(x)[idx], np.array(y)[idx]

X_train, Y_train, X_valid, Y_valid = split_valid_set(x, y, 0.9)

Y_train = Y_train.ravel()
Y_valid = Y_valid.ravel()
# X_train = np.array(x[:29000,:])
# Y_train = np.array(y[:29000,:]).ravel()


#X_train = (X_train-np.min(X_train, axis=0))/np.max(X_train, axis=0)


# X_test = np.array(x[29000:,:])
# X_test = (X_test-np.min(X_train, axis=0))/np.max(X_train, axis=0)
# Y_test = np.array(y[29000:,:]).ravel()


gbm = xgb.XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.05).fit(X_train, Y_train)
# predictions = gbm.predict(x_test)


#model.fit(X_train, Y_train)
acc = sum(np.round(gbm.predict(X_valid))==Y_valid)/float(len(Y_valid))
print ( sum(np.round(gbm.predict(X_valid))==Y_valid)/float(len(Y_valid)) )


#write_file(np.rint(gbm.predict(X_test)), 'answer_'+str(acc)+'_xgb.csv')
joblib.dump(gbm, 'model_'+str(acc)+'.pickle') 
#gbm.get_booster().dump_model('model_'+str(acc)+'_.model')


