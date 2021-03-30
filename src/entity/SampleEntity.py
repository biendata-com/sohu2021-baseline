# _*_ coding:utf-8 _*_
'''
@author:   zhangfeng
@mail:  fengzhang217463@sohu-inc.com
@date:  2021/3/16 上午11:29
@file: SampleEntity
'''

import json

class Sample(object):
    def __init__(self, source:str, target:str, label:int):
        self.source_text = source
        self.target_text = target
        self.label = label

    @staticmethod
    def load(s:str):
        tmp = json.loads(s)
        if "source" not in tmp or "target" not in tmp or ("labelA" not in tmp and "labelB" not in tmp):
            raise ValueError("sample is incomplete")
        if "labelA" in tmp:
            label = int(tmp["labelA"])
        else:
            label = int(tmp["labelB"])
        source = tmp["source"]
        target = tmp["target"]
        sample = Sample(source, target, label)
        return sample

