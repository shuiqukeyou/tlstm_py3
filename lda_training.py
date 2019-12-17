import pickle

from gensim.models import LdaModel
from gensim import corpora, models

from dataload import load_data


def train_lda(dataset):
    new_brs2 = []
    for br in dataset:
        new_brs2.append([str(a) for a in br])

    # 创建语料的词语词典，每个单独的词语都会被赋予一个索引
    dic = corpora.Dictionary(dataset)
    common_corpus = [dic.doc2bow(text) for text in new_brs2]

    lda_model = models.ldamulticore.LdaMulticore(common_corpus, num_topics=100)
    # 保存lda模型
    lda_model.save("lda_model")

    # 保存生成的字典
    with open("dictionary.b", "wb+") as f:
        pickle.dump(dic, f)

if __name__ == '__main__':
    docs, _ = load_data("data_ma.npz", 40, 25000, 700)
    train_lda(docs)
    with open("dictionary.b", "rb") as f:
        dic = pickle.load(f)
    lda_model = LdaModel.load("lda_model")
    other_corpus = [dic.doc2bow(text) for text in docs]
    temp = [dic.doc2bow(text) for text in docs[:10]]
    t = []
    for br in temp:
        t.append(lda_model.get_document_topics(br, minimum_probability=0))
    for i in t:
        s = [w[1] for w in i]
        print(s)
