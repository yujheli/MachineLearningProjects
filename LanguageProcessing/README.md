ML 2017 Fall Final - TV Conversation
========================

Used Library:
------------------------

- gensim 3.2.0
- numpy
- pandas
- jieba (TW)

## Install 3.2.0 gensim 

```
    pip3 intall gensim==3.2.0
    
```
or
```
    pip3 intall gensim --upgrade
    
```
to install newest gensim 

Note that if you install 3.1.0 or older version, you may encounter errors when testing


## Download used models and traditional jieba


1. Run the following command to download **the Kaggle dataset** and **the Traditional Chinese version of jieba**.

```
    bash Download.sh
```

Note that if those things have been there, the script will **not** download them again.
 
Note:
------------------------
1. Some important helping functions are written in **utils.py**, so this file should be preserved carefully.
User guide:
------------------------

## Testing
If you want to run the testing code

```bash

    bash  run_test.sh  <testing csv file>  <prediction csv file>

```

## Training

If you want to run the training code, run the following command to produce our **dictionary** and the **embedding layer** in **src** folder.

```
    bash run_train.sh arg1

```

where **arg1** denotes how many sentences we want to combine into a unit when training word2vecs. If those things have been there, you can skip this step directly.



