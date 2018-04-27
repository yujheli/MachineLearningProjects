import numpy as np
import pandas as pd
import sys
#training_dataset = pd.read_csv("train.csv",  encoding='big5')

def write_file(out_list, filename):
    with open(filename,'w') as output:
        output.write('id,value')
        counter = 0
        for i in out_list:
            output.write('\n')
            output.write('id_'+str(counter))
            output.write(',')
            output.write(str(i))
            counter+=1

def toValue(x):
    if x =='NR':
        return 0.0
    else:
        return float(x)

def extract_important_features(dataset, is_test=False):

    #selected = [2 , 7, 8 ,9, 11 ,12]
    selected = [9]


    window_size = 9
    ext = []
    for i in dataset:
        xx = []
        for j in selected:
            #print (list(i[9*j+9-window_size:9*j+9]))
            xx.extend(list(i[9*j+9-window_size:9*j+9]))
        #print(xx)
        if is_test == False:
            xx.append(i[-1])
        ext.append(xx)
    return ext


testing_dataset = pd.read_csv(sys.argv[1],  encoding='big5', header=None)

test = testing_dataset.iloc[:,2:].applymap(toValue)
#print (test)
testing_dataset = np.array(test).flatten().reshape((240,162))

test_set = np.array(extract_important_features(testing_dataset,is_test=True))

test_x = np.concatenate((np.ones((test_set.shape[0],1)), test_set), axis = 1)

theta = np.load('model.npy')
predictions = test_x.dot(theta[0]) + (test_x**2).dot(theta[1])

write_file(predictions,sys.argv[2])












