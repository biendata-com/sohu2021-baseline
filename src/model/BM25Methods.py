# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 下午3:44
@file: BM25Methods
'''

from typing import List
from src.entity.SampleEntity import Sample
from gensim.summarization.bm25 import BM25


def train_bm25Model(Dataset: List[Sample], Stopwords: set) -> BM25:
    corpus = []
    for cur_sample in Dataset:
        corpus.append([word for word in cur_sample.source_text.split(" ") if word not in Stopwords])
        corpus.append([word for word in cur_sample.target_text.split(" ") if word not in Stopwords])
    bm25Model = BM25(corpus)
    return bm25Model


def calculate_bm25_similarity(bm25Model: BM25, tokens1: List[str], tokens2: List[str]) -> float:
    token_sequence1, token_sequence2 = (tokens1, tokens2) if len(tokens1) > len(tokens2) else (tokens2, tokens1)

    word_sequences = {}
    # todo: 为什么只计算token_sequence2的词频
    for word in token_sequence2:
        if word not in word_sequences:
            word_sequences[word] = 0
        word_sequences[word] += 1

    param_k1 = 1.5
    param_b = 0.75

    score1, score2 = 0.0, 0.0
    for word in token_sequence1:
        if word not in word_sequences or word not in bm25Model.idf:
            continue
        score1 += (bm25Model.idf[word] * word_sequences[word] * (param_k1 + 1) / (
                word_sequences[word] + param_k1 * (1 - param_b + param_b * 1)))
    for word in token_sequence2:
        if word not in word_sequences or word not in bm25Model.idf:
            continue
        score2 += (bm25Model.idf[word] * word_sequences[word] * (param_k1 + 1) / (
                word_sequences[word] + param_k1 * (1 - param_b + param_b * 1)))

    similarity = score1 / score2 if score2 > 0 else 0
    similarity = min(similarity, 1.0)
    return similarity
