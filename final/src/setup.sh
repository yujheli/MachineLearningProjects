#!/bin/bash

if ! [[ -e ./jieba ]]; then
    git clone https://github.com/ldkrsi/jieba-zh_TW.git
    mv ./jieba-zh_TW/jieba ./jieba
    rm -rf ./jieba-zh_TW
fi
