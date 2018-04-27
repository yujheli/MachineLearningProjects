import numpy as np
import pandas as pd
training_dataset = pd.read_csv("train.csv",  encoding='big5')
#testing_dataset = pd.read_csv("test.csv",  encoding='big5', header=None)



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

train = training_dataset.iloc[:,3:].applymap(toValue)
frames = [train[i * 18 : i * 18 + 18].reset_index(drop = True) for i in range(int(len(train) / 18))]
df = pd.concat(frames, axis=1)


# Generating Training Dataset
data_list = []
for i in range(len(df.columns)):
    if i+9 >= len(df.columns):
        break
    else:
        
        xx = list(np.array(df.iloc[:,i:i+9]).flatten())
        xx.append(df.iloc[9,i+9])
        data_list.append(xx)

train_set = np.array(extract_important_features(data_list))

#training parameters

iterations = 1000000
l_rate = 0.1
lambda_ = 0.1

#start machine learning training

X = np.array(train_set[:,:-1])
X_2 = X ** 2
y = np.array(train_set[:,-1])


x = np.concatenate((np.ones((X.shape[0],1)), X), axis = 1)
x_2 = np.concatenate((np.ones((X_2.shape[0],1)), X_2), axis = 1)

#theta = np.load('model.npy')
theta = np.zeros((2,X.shape[1]+1))
#theta.append(np.zeros((X.shape[1]+1)))
#theta.append(np.zeros((X.shape[1]+1)))

x_t = x.transpose()
x_t2 = x_2.transpose()
s_gra = np.zeros(len(x[0]))
s_gra2 = np.zeros(len(x[0]))


for i in range(iterations):
      
        predictions = x.dot(theta[0]) + x_2.dot(theta[1])
        loss = predictions - y
        
        cost = np.sum(loss**2) / len(x)
        cost_a  = np.sqrt(cost)

        gra = x_t.dot(loss) + lambda_ * (theta[0])      
        gra2 = x_t2.dot(loss) + lambda_ * (theta[1])

        s_gra += gra**2
        s_gra2 += gra2**2
        ada = np.sqrt(s_gra)
        ada2 = np.sqrt(s_gra2)

        theta[0] -= l_rate * gra/ada
        theta[1] -= l_rate * gra2/ada2

        print ('iteration: %d | Cost: %f ' % ( i,cost_a))
        

            #np.save('model.npy',theta)


# save model
np.save('model.npy',theta)
# read model








