#!/bin/bash
cd src
if ! [[ -e ./provideData  ]]; then
    wget https://www.dropbox.com/s/x7m7v6dbhzv9ic1/provideData.zip?dl=1 -O provideData.zip
    unzip provideData.zip
    rm provideData.zip
fi

python3 build_word2vec.py $1
