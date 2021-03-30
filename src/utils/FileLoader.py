# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 上午11:31
@file: FileLoader
'''

import os
from typing import List
from src.entity.SampleEntity import Sample

def load_stopwords_from_file(stopwords_file:str=None) -> set:
    if stopwords_file is None:
        return set()
    if not os.path.exists(stopwords_file):
        raise ValueError("stopwords_file: {} doesn't not exist".format(stopwords_file))

    stopwords = set()
    with open(stopwords_file,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            word = line.strip()
            stopwords.add(word)
    return stopwords

def load_dataset_from_file(input_file:str) -> List[Sample]:
    if not os.path.exists(input_file):
        raise ValueError("input file: {} doesn't not exist".format(input_file))
    with open(input_file,'r') as fr:
        lines = fr.readlines()
        return [Sample.load(x) for x in lines]
