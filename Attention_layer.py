from keras import backend as K
from keras.engine.topology import Layer

class AttentionLayer(Layer):
    def __init__(self, units, **kwargs):
        # 官方文档要求调用
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.input_dim_lstm = 0
        self.input_dim_top = 0
        self.input_en_times = 0

    def build(self, input_shape):
        # input_shape:[(None, 40, 500), (None, 100)]
        self.input_dim_lstm = input_shape[0][-1]  # 编码器的宽度,500
        self.input_en_times = input_shape[0][-2]  # 编码器的条目数,40
        self.input_dim_top = input_shape[1][-1]  # 主题向量的的宽度,100


        # 编码器隐藏值权重(500*500)
        self.w_lstm = self.add_weight(name='w_en', shape=(self.input_dim_lstm, self.units),
                                      initializer='glorot_uniform', trainable=True)
        # 主题值权重(100*500)
        self.w_top = self.add_weight(name='w_de', shape=(self.input_dim_top, self.units),
                                     initializer='glorot_uniform', trainable=True)
        # u(500*1)
        self.nu = self.add_weight(name='nu', shape=(self.units, 1),
                                  initializer='glorot_uniform', trainable=True)
        # 官方文档要求调用
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        # 读取编码隐藏序列、主题序列
        lstm_seq = inputs[0]  # 输入值为 batch * time_step * LSTM_size，理论上是100 * 40 * 500
        top_seq = inputs[1]  # batch * lda_size，理论上是100 * 100

        # 40*500 * 500*500 = 40*500
        # K.reshape(lstm_seq, (-1, self.input_dim_lstm)  # batch*time_step * LSTM_size
        # (batch*time_step * LSTM_size) * (LSTM_size * att_size) = batch*time_step  * att_size
        att_en = K.dot(K.reshape(lstm_seq, (-1, self.input_dim_lstm)), self.w_lstm)  # (40*batch * 500)
        att_en = K.reshape(att_en, shape=(-1, self.input_en_times, self.units))  # batch * 40 * 500

        # ?*100 * 100*500 = ?*500
        # (batch * lda_size） * （lad_size * att_size） = batch * att_size
        att_top = K.dot(top_seq, self.w_top)
        # 添加一个维度
        att_top = K.reshape(att_top, (-1, 1, self.units))  # ? * 1 * 256
        # 复制为和编码序列相同
        att_top = K.repeat_elements(att_top, self.input_en_times, 1)  # ? * 40 * 256

        co_m = att_en + att_top  # ? * 40 *500
        co_m = K.reshape(co_m, (-1, self.units))  # ?*40 * 500
        alpha = K.tanh(co_m)

        # uij
        mu = K.dot(alpha, self.nu)  # ?*40 * 1
        mu = K.reshape(mu, (-1, self.input_en_times))  # ? * 1 * 40

        # 注意力权重
        alphas = K.softmax(mu)  # ? * 1 * 40
        sum_en = K.sum(lstm_seq * K.expand_dims(alphas, -1), 1)
        return sum_en

    # 声明支持mask后必须有这个函数
    def compute_mask(self, inputs, mask=None):
        return None

    # 供框架调用，用于推断输出的形状
    def compute_output_shape(self, input_shape):
        # 输出尺寸：batch_size * lstm_size
        output_shape = (input_shape[1][0], input_shape[0][-1])
        return output_shape
