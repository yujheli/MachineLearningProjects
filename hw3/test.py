

import pandas as pd
import numpy  as np
import sys
from keras.models import Sequential, load_model
from keras.layers import *

def write_file(out_list, filename):
    with open(filename,'w') as output:
        output.write('id,label')
        counter = 0
        for i in out_list:
            output.write('\n')
            output.write(str(counter))
            output.write(',')
            output.write(str(int(i)))
            counter+=1

model = load_model(sys.argv[3])
model.summary()

mean = np.load("mean_value.npy")
std = np.load("mean_std.npy")

test = pd.read_csv(sys.argv[1])

def splt(s):
    return [float(i)/255 for i in s.split(' ')]

data = list(test['feature'].apply(splt))

t = np.array(data)
t -= mean
t/= std

x_test = t.reshape(t.shape[0],48,48,1)


label = model.predict(x_test)
write_file(np.argmax(label,axis=1),sys.argv[2])




