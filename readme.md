## 介绍
论文**Hashtag recommendation with topical attentionbased LSTM**的代码实现，做baseline用，所以实现出来。

没有官方源码（至少没有一个完整能用的），所以可能和论文的思想/结果有所出入


## 需求
开发时使用的相关库的版本，仅供参考，不是最低或最高版本需求

- python == 3.7
- Numpy == 1.17.3 
- Tensorflow == 1.15.0
- Keras == 2.24
- gensim == 3.8

## 文件结构
- main.py：主文件
- Attention_layer.py：注意力层
- dataload.py：数据装载函数
- function.py：精确度、召回、成本函数等
- lda_training.py:训练LDA用
- word2vec_training.py：训练word2vec用

## 注意事项
需要先训练LDA和word2vec

不一定能和官方的结果一样

lad和word2vec数据的装载模式是全部预装载在内存中再进行训练，非常的耗内存，实际上应该做一个不需要训练的中间层处理数据，但是我懒，炸内存建议自己改一下

