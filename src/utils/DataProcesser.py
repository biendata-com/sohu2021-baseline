# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 下午3:11
@file: DataProcesser
'''

import jieba
from typing import List
from tqdm import tqdm
from gensim.summarization.bm25 import BM25
from src.entity.SampleEntity import Sample
from src.model.BM25Methods import calculate_bm25_similarity
from src.model.TfidfMethods import calculate_tf_cosine_similarity,calculate_tfidf_cosine_similarity
import numpy as np

def seg_dataset(Dataset:List[Sample]) -> None:
    print("take a break and be patient with the segmentation")
    for sample in tqdm(Dataset):
        sample.source_text = " ".join(jieba.cut(sample.source_text))
        sample.target_text = " ".join(jieba.cut(sample.target_text))
    return

def make_text_pair_features(tokens1:List[str], tokens2:List[str], BM25Model:BM25) -> List[float]:
    features = []
    features.append(calculate_bm25_similarity(BM25Model, tokens1, tokens2))  # BM25的相似值

    features.append(calculate_tf_cosine_similarity(tokens1,tokens2))  # TF的余弦相似度
    features.append(calculate_tfidf_cosine_similarity(tokens1,tokens2, BM25Model.idf)) # TF-IDF的余弦相似度
    # todo: use more text distance as features
    # features.append(jaccard_common_words(text1, text2))
    # features.append(ochiai_common_words(text1, text2))
    return features

def dataset_features_process(Dataset:List[Sample], BM25Model:BM25) -> (np.ndarray, np.ndarray):
    features = [make_text_pair_features(x.source_text.split(" "), x.target_text.split(" "), BM25Model) for x in Dataset]
    labels = [x.label for x in Dataset]
    return np.array(features), np.array(labels)