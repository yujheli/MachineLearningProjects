
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
from sklearn.externals import joblib


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



X_test = np.array(pd.read_csv(sys.argv[1]).values)



# gbm = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)
# predictions = gbm.predict(x_test)

clf = xgb.XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05)
clf = joblib.load(sys.argv[3]) 


write_file(np.rint(clf.predict(X_test)), sys.argv[2])


