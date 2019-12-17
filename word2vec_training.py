import numpy as np

from gensim.models import Word2Vec

from dataload import load_data


def train_word2vec(data):
    model = Word2Vec(data, sg=1, size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")

if __name__ == '__main__':
    docs, _ = load_data("data_ma.npz", 40, 25000, 700)
    train_word2vec(docs)
    model2 = Word2Vec.load("word2vec.model")
    temp = model2.wv[docs[0]]
    temp = temp[:20]
    print(temp.shape)
    temp2 = np.zeros((100,10)).T
    print(temp2.shape)
    temp = np.insert(temp2, 0, values=temp, axis=0)
    print(temp.shape)


    # [model2[text] for text in docs]
    # print(temp)