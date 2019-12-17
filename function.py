import numpy as np
from keras import backend as K

# tag one-hot化
def vec2one_hot(tags, vec_size):
    value = tags[0]
    temp = np.zeros(vec_size, dtype=np.int16)  # 省内存
    temp[value] = 1
    return temp
    # one_hots = []
    # for value in tags:
    #     temp = np.zeros(vec_size, dtype=np.int16)  # 省内存
    #     temp[value] = 1
    #     one_hots.append(temp)
    # return one_hots

# tag向量化
def tag2vec(tags, vec_size):
    one_hots = np.zeros(vec_size, dtype=np.int16)  # 省内存
    for value in tags:
        one_hots[int(value)] = 1
    return one_hots

# 损失函数（交叉损失）
def mean_negative_log_probs(y_true, y_pred):
    log_probs = -K.log(y_pred)
    log_probs *= y_true
    # return K.sum(log_probs)
    return K.sum(log_probs) / K.sum(y_true)


# 损失函数（二元交叉熵）
def mean_binary_crossentropy(y_true, y_pred):
    temp = y_true * K.log(y_pred + K.epsilon()) + (1-y_true) * K.log(1-y_pred+K.epsilon())
    return K.sum(-temp) / K.sum(y_true)

# 准确度函数
def compute_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # 计算真值1且预测1
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))  # 预测总数
    precision = true_positives / (predicted_positives + K.epsilon())  # K.epsilon()：极小量
    return precision


# 召回率计算
def compute_recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # 计算真值1且预测1
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))  # 真值总数
    recall = true_positives / (possible_positives + K.epsilon())
    return recall