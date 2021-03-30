# 欢迎参加SOHU2021文本匹配算法大赛！

[报名网站](https://www.sohu.com/) | [比赛论坛](https://www.sohu.com/) | [了解搜狐](https://www.sohu.com/)

## 比赛Baseline
本项目是sohu2021-文本匹配算法大赛的baseline，基于bm25和多层感知机（MLP）分类器完成

#### 环境搭建
```bash
pip install -r requirements.txt
```
#### 快速上手
```bash
python train_and_evaluate.py --input_file data/raw/samples.txt 
```

#### 数据格式
```json
{
  "source": "日媒：国际奥委会考虑将参加东京奥运会开幕式运动员减半", 
  "target": "东京奥运会开幕式或将缩小规模", 
  "labelA": "1"
}
```

## 引用

## Licence

## Reference
https://www.sohu.com/