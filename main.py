import numpy as np
import pickle
import os
from decimal import Decimal

from gensim.models import LdaModel, Word2Vec
from keras.models import Model
from keras.layers import Input, Dense, Dropout, GRU

# 模型变量储存、早终止
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Attention_layer import AttentionLayer
from dataload import load_data

from function import tag2vec, mean_negative_log_probs, compute_precision, compute_recall

# 最多40个词
MAX_WORD = 40
ALL_WORDS = 25000
ALL_TAGS = 700
LDA_SIZE = 100
EMBEDDING_DIM = 100
PRE_SIZE = 1
PART = 0.1
NUM_EPOCHS = 60
BATCH_SIZE = 128

LSTM_SIZE = 500
ATTENTION_SIZE = 250

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 输入层
wordvec_input = Input(shape=(MAX_WORD, EMBEDDING_DIM))
lda_input = Input(shape=(LDA_SIZE,))

# LSTM
# 每个单元的隐藏状态是1X500，最多输入40个单词，即输出为40 * 500
# 此处将LSTM单元改为了GRU单元
encoder_lstm = GRU(LSTM_SIZE, return_sequences=True, return_state=True, kernel_initializer='orthogonal',
                   recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=0.1)
lstm_seq,  _ = encoder_lstm(wordvec_input)

# 注意力
attention = AttentionLayer(units=ATTENTION_SIZE)
output = attention([lstm_seq, lda_input])

drop = Dropout(PART)
output = drop(output)
# 全连接/sigmoid层
weight_dense = Dense(ALL_TAGS, activation='softmax')  # 2500
tag_out = weight_dense(output)

model = Model(inputs=[wordvec_input, lda_input], outputs=[tag_out])
# compile model
model.compile(optimizer='adam', loss=mean_negative_log_probs, metrics=[compute_precision, compute_recall])
model.summary()

# 装载数据
new_brs, sfs = load_data(path="data_ma.npz", lenth=40, num_words=25000, num_sfs=700, per=0.8)

# new_brs = new_brs[:200]
# sfs = sfs[:200]

sfs_back = sfs[:]
# 对tag进行裁切，只保留每篇文章的前x个
sfs = [sf[:1] for sf in sfs]

# 加载字典文件
with open("dictionary.b", "rb") as f:
    dic = pickle.load(f)
# 加载LDA模型
lda_model = LdaModel.load("lda_model")

# 转换为dict字典编号
new_brs_dic = [dic.doc2bow(text) for text in new_brs]

# 生成LDA向量
lad_ver = []
for a in new_brs_dic:
    temp = lda_model.get_document_topics(a)
    t2 = np.zeros(LDA_SIZE)
    for d in temp:
        t2[d[0]] = d[1]
    lad_ver.append(t2)

# one-hot化:[0,1,0,0,...,1,0,0,1]
sfs_one = [tag2vec(sf, ALL_TAGS) for sf in sfs]

# 加载word2vec模型
w2v_model = Word2Vec.load("word2vec.model")
doc_w2v = [w2v_model[text][:40] for text in new_brs]  # 超过40切到40
doc_w2v_ = []
for vec in doc_w2v:
    t = 40 - len(vec)
    if t > 0:
        temp = np.zeros((t, 100))  # 不足40补0
        vec = np.insert(temp, 0, values=vec, axis=0)
    doc_w2v_.append(vec)
doc_w2v = doc_w2v_

lad_ver = np.array(lad_ver)
sfs_one = np.array(sfs_one)

split = int(len(sfs)*PART)

brs_train = doc_w2v[:-split]
sfs_train = sfs_one[:-split]
lad_train = lad_ver[:-split]

brs_test = doc_w2v[-split:]
sfs_test = sfs_back[-split:]
lad_test = lad_ver[-split:]

es = EarlyStopping(monitor='val_loss', patience=2)
cp = ModelCheckpoint(filepath='GRU_sigmoid.h5', monitor='val_loss', save_best_only=True)

model.fit([brs_train, lad_train], [sfs_train], validation_split=0.1, epochs=NUM_EPOCHS,
          batch_size=BATCH_SIZE, callbacks=[es, cp], verbose=2)

model.load_weights('GRU_sigmoid.h5')  # 加载最好的训练结果
# 开始预测
pre_model = Model(inputs=[wordvec_input, lda_input], outputs=[tag_out])


def pre_list(va=0.5):
    temp1 = 0
    temp2 = 0
    temp3 = 0

    for i in range(len(brs_test)):
        s1 = np.array([brs_test[i]])
        s2 = np.array([lad_test[i]])
        pre_seq = pre_model.predict([s1, s2])
        pre_seq = list(pre_seq[0])
        tru_seq = sfs_test[i]

        # 如果仅预测1个tag，按照一般的softmax的判断方法，阙值默认取0.5
        if PRE_SIZE == 1:
            pre_seq2 = []
            for index in range(len(pre_seq)):
                if pre_seq[index] > va:
                    pre_seq2.append(index)
                    break
            # 取输出序列和测试集的交集
            intersection = list(set(tru_seq).intersection(set(pre_seq2)))
            temp1 += len(intersection)
            temp2 += len(pre_seq2)
            temp3 += len(tru_seq)
        else:  # 如果需要预测多个tag，从高到低去前x个
            pre_seq = list(zip(range(len(pre_seq)), pre_seq))
            pre_seq.sort(key=lambda r: r[1], reverse=True)
            pre_seq = [tag[0] for tag in pre_seq[:PRE_SIZE]]

            # 取输出序列和测试集的交集
            intersection = list(set(tru_seq).intersection(set(pre_seq)))
            # print(intersection, pre_seq, tru_seq)
            temp1 += len(intersection)
            temp2 += len(pre_seq)
            temp3 += len(tru_seq)

    print(str(Decimal(va).quantize(Decimal('0.00'))))
    precision2 = temp1 / temp2
    recall2 = temp1 / temp3
    f1_2 = 2 * precision2 * recall2 / (precision2 + recall2)
    print("precision_2:" + str(Decimal(precision2).quantize(Decimal('0.000'))) + " ; " +
          "recall_2:" + str(Decimal(recall2).quantize(Decimal('0.000'))) + " ; " +
          "f1_1:" + str(Decimal(f1_2).quantize(Decimal('0.000'))))


# l = [0.05 * x for x in range(20)[2:11]]
# for i in l:
#     pre_list(i)
pre_list(0.5)
