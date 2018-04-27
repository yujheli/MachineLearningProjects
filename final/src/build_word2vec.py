import os
import sys
import json
import numpy as np
import utils as ut # own-made functions
from gensim.models import Word2Vec

# check arguments
if len(sys.argv) < 2:
    print('usage: python3 %s arg1' % sys.argv[0])
    print('arg1: number_of_sentences_as_a_unit')
    exit(0)

# global configurations
feature_size = 384
unit = int(sys.argv[1])

# specify data directory
train_data_dir = './provideData/training_data'
test_data_dir = './provideData'
output_dir = './model'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# read data into sentences
sentences = []
for i in range(1, 6):
    sen_list = ut.read_sentence_train(train_data_dir + '/' + str(i) + '_train.txt', unit)
    sentences += sen_list

# build our word2vec and save it
word2vec = Word2Vec(sentences, size=feature_size, workers=4, min_count=0, sg=1, window=100)
np.save(output_dir + '/word2vec_' + sys.argv[1] + '.npy', word2vec)
