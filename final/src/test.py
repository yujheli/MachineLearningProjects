import os
import sys
import json
import numpy as np
# import logging
import jieba
import re
import json

import numpy as np
from utils import * 
from gensim.models import Word2Vec

# sys.path.append('../')
# logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
# logging.info("running %s" % " ".join(sys.argv))
# check arguments
if len(sys.argv) < 2:
    print('usage: python3 %s arg1' % sys.argv[0])
    print('arg1: output of predicting csv file')
    exit(0)

# global configurations
feature_size = 64


# specify data directory
# train_data_dir = './provideData/training_data'
# test_data_dir = './provideData'
# output_dir = './model'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)



print('Loading word2vec model')
word2vec = np.load('word2vec_no1.npy')[()]

# produce the testing result
result_option = []
with open('../'+sys.argv[1], 'r') as fr:
    # ignore the header
    next(fr)

    # read dialogue and answers
    for line_ori in fr:

        # read dialogue (question)
        line = line_ori
        line = line.strip().split(',')[1].replace("\t", " ")
        line = re.split(':| |A|B|C', line)
        sens = list(filter(None, line)) # remove all empty strings

        # calculate dialogue vector
        emb_cnt = 0
        avg_dlg_emb = np.zeros((feature_size,))
        for sen in sens:
            for word in list(jieba.cut(sen)):
                try:
                    avg_dlg_emb += word2vec[word]
                    emb_cnt += 1
                except KeyError:
                    continue
        if emb_cnt > 0:
            avg_dlg_emb /= emb_cnt

        # want to choose the best option
        idx = -1
        max_idx = -1
        max_sim = -10 #if set with -5 is better
        answers = line_ori.strip().split(',')[2]
        for ans in answers.strip().split(':')[1:]: # process one option for each iteration
            idx += 1

            # read options (answer)
            ans = ans.strip().replace("\t", " ")
            sens = re.split('A|B|C| ', ans)
            sens = list(filter(None, sens)) # remove all empty strings

            # calculate answer vectors
            # 在六個回答中，每個答句都取詞向量平均作為向量表示
            # 我們選出與dialogue句子向量表示cosine similarity最高的短句
            emb_cnt = 0
            avg_ans_emb = np.zeros((feature_size,))
            for sen in sens:
                for word in jieba.cut(sen):
                    try:
                        avg_ans_emb += word2vec[word]
                        emb_cnt += 1
                    except KeyError:
                        continue
            if emb_cnt > 0:
                avg_ans_emb /= emb_cnt
            if np.linalg.norm(avg_dlg_emb) > 0 and np.linalg.norm(avg_ans_emb) > 0:
                sim = np.dot(avg_dlg_emb, avg_ans_emb) / np.linalg.norm(avg_dlg_emb) / np.linalg.norm(avg_ans_emb)
            else:
                sim = 0
            if sim > max_sim:
                max_idx = idx
                max_sim = sim
        
        # save answers to a temporary list
        result_option.append(max_idx)

# output our answer to file



print("Starting ensembling")
import sys
import numpy as np
import pandas as pd


model2 = np.load('word2vec_no2.npy')
result_option = ensemble(result_option, model2)  

with open('../'+sys.argv[2], 'w') as fw:
    print('id,ans', file=fw)
    for idx, result in enumerate(result_option):
        print('%d,%d' % (idx+1, result), file=fw)
