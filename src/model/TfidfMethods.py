# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 下午6:10
@file: TfidfMethods
'''

from typing import List
from scipy.linalg import norm

def gen_tf(tokens:List[str]) -> dict:
    '''
    TF(term frequency) is defined as:
        tf(token) = Frequency(token) / Count(all_tokens)
    :param tokens: List[str]
    :return: Dict
    '''
    total = len(tokens)
    tf_dict = {}
    for word in tokens:
        tf_dict[word] = tf_dict.get(word, 0.0) + 1.0
    for word in tf_dict:
        tf_dict[word] /= total
    return tf_dict


def gen_tfidf(tokens:List[str], idf_dict:dict) -> dict:
    '''
    Tf-Idf(term frequency–inverse document frequency) is defined as:
        tf(token) = Frequency(token) / Count(all_tokens)
        idf(token) is implemented by querying the bm25 model, whose building function is:
            idf(token) = log( (Count(all_docs) - Count(contain_token_docs) + 0.5) / (Count(contain_token_docs) + 0.5) )
        tf-idf(token) = tf(token) * idf(token)
    :param tokens: List[str]
    :param idf_dict: Dict[(str,float)]
    :return:
    '''
    total = len(tokens)
    tfidf_dict = {}
    for w in tokens:
        tfidf_dict[w] = tfidf_dict.get(w, 0.0) + 1.0
    for k in tfidf_dict:
        tfidf_dict[k] *= idf_dict.get(k, 0.0) / total
    return tfidf_dict


def cosine_similarity(a:dict, b:dict) -> float:
    longer_dict, shorter_dict = (a, b) if (len(a) > len(b)) else (b, a)
    res = 0
    for (word_name, word_value) in longer_dict.items():
        res += word_value * shorter_dict.get(word_name, 0)
    if res == 0:
        return 0.
    try:
        res = res / (norm(list(longer_dict.values())) * norm(list(shorter_dict.values())))
    except ZeroDivisionError:
        res = 0.
    return res

def calculate_tf_cosine_similarity(tokens1:List[str], tokens2:List[str]) -> float:
    tf1 = gen_tf(tokens1)
    tf2 = gen_tf(tokens2)
    return cosine_similarity(tf1, tf2)


def calculate_tfidf_cosine_similarity(tokens1:List[str], tokens2:List[str], idf_dict:dict) -> float:
    tfidf1 = gen_tfidf(tokens1, idf_dict)
    tfidf2 = gen_tfidf(tokens2, idf_dict)
    return cosine_similarity(tfidf1, tfidf2)