
import pandas as pd
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.externals import joblib

raw_data = []
counter = 0


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



# with open('training_label.txt','rt') as inputfile:
#     for line in inputfile.read().split('\n'):
#         raw_data.append(line.split(' +++$+++ '))

# raw_data = raw_data[:-1]
# data = pd.DataFrame(raw_data)

# word_size = 20000
# t = Tokenizer(num_words=word_size)
# #t = Tokenizer()

# t.fit_on_texts(data.iloc[:,1])
# vocab_size = word_size + 1
# integer encode the documents
t = joblib.load('pun_tokenizer.pkl')


test_data = []
count = 0
with open(sys.argv[1],'rt') as inputfile:
    for line in inputfile.read().split('\n'):
        if count==0:
            count+=1
        else:
            xx = []
            tmp = line.split(',')
            xx.append(tmp[0])
            xx.append(','.join(tmp[1:]))
            test_data.append(xx)
            
test_data = test_data[:-1]   

test = pd.DataFrame(test_data)

encoded_docs = t.texts_to_sequences(test.iloc[:,1])


max_length = 36
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

count = 0
print("model length: ", len(sys.argv)-3)
for i in range(len(sys.argv)-3):
    model = load_model(sys.argv[i+3])
    model.summary()
    score = model.predict(padded_docs,verbose=1)

    if i == 0:
        result = score
        count+=1.0
    else:
        result += score
        count+=1.0

result = np.round(result/count)
write_file(result,sys.argv[2])






