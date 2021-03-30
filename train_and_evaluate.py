# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 上午10:47
@file: train_and_evaluate
'''

import os
import json
import argparse
from src.utils.FileLoader import load_stopwords_from_file,load_dataset_from_file
from src.utils.DataProcesser import seg_dataset,dataset_features_process
from src.model.BM25Methods import train_bm25Model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle
import joblib

def main(args) -> None:
    '''

    :param args: args map
    :return:None
    1. 加载文本并分词
    2. 训练bm25模型
    3. 利用bm25模型做feature
    4. 训练mlp分类器
    5. 评估
    '''

    # 加载数据集
    dataset = load_dataset_from_file(args.input_file)

    # 分词
    seg_dataset(dataset)

    # 加载停用词表
    Stopwords = load_stopwords_from_file(args.stopwords_file)

    # 准备bm25
    bm25Model = train_bm25Model(dataset, Stopwords)

    # 特征转换
    features, labels = dataset_features_process(dataset,bm25Model)

    # 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # 训练MLP分类器
    clf = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam',
                        learning_rate_init=1e-4, max_iter=1000)
    clf.fit(X_train, y_train)

    # 评估
    predictions = clf.predict(X_test)
    print('准确率：%s' % clf.score(X_test, y_test))
    print('交叉验证准确率：%s' % cross_val_score(clf, X_test, y_test, cv=5).mean())
    print('F1-score: %s' % f1_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    # 保存
    if args.model_path is not None:
        os.makedirs(args.model_path,exist_ok=True)
        with open(os.path.join(args.model_path,"bm25.pkl"),'wb') as bmfw:
            pickle.dump(bm25Model,bmfw)
        joblib.dump(clf, os.path.join(args.model_path, "mlp.pkl"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="训练集文件位置")
    parser.add_argument("--stopwords_file", type=str, required=False, help="停用词文件位置")
    parser.add_argument("--model_path", type=str, required=False, default=None, help="训练模型保存位置")
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4, ensure_ascii=False))
    main(args)
