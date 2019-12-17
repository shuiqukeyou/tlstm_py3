import numpy as np


def load_data(path, lenth, num_words=None,
              num_sfs=None, per=1):

    with np.load(path, allow_pickle=True) as f:
        brs = f['brs']  # 文章
        sfs = f['sfs']  # tag
        # brs = f['bodies']  # 文章
        # sfs = f['tags']  # tag

    # 防止炸内存
    brs = brs[:int(len(brs) * per)]
    sfs = sfs[:int(len(sfs) * per)]
    indices = np.arange(len(brs))  # 按文章篇数返回一个array对象：0，1，2...len(brs)-1
    # np.random.seed(114)
    np.random.shuffle(indices)  # 随机打乱array对象顺序

    # 将文本、tag统一打乱
    brs = brs[indices]
    sfs = sfs[indices]

    # 如果没有设定词汇编号上限，则读取所有文章，取其中编号最大的词汇作为上限
    if not num_words:
        num_words = max([max(x) for x in brs])
    # 如果没有设定tag编号上限，则读取所有文章，取其中编号最大的tag作为上限
    if not num_sfs:
        num_sfs = max([max(x) for x in sfs])

    # 遍历每篇文章
    # 遍历每个文章的每个词，如果某个词的编号大于设定词汇数量上限，将其抛弃
    _docs = []
    for x in brs:
        temp = []
        for w in x:
            if w < num_words:
                temp.append(str(w))
        _docs.append(temp)
    brs = _docs


    # filter(判断函数,可迭代对象)：只保留迭代对象中满足判断函数的值
    # py2返回list，py3返回迭代器
    sfs = [filter(lambda x: x < num_sfs, sf) for sf in sfs]  # 排除所有大于tag编号上限的tag
    sfs = [list(sf) for sf in sfs]

    docs = []
    tags = []
    for doc, tag in zip(brs, sfs):
        if len(doc) == 0 or len(sfs) == 0:
            continue
        else:
            docs.append(doc)
            tags.append(tag[:lenth])

    doc_counts = [len(doc) for doc in docs]
    doc_counts = sum(doc_counts)

    doc_tags = [len(tag) for tag in tags]
    doc_tags = sum(doc_tags)

    print("even_word:" + str(doc_counts/len(docs)))
    print("even_tag:" + str(doc_tags / len(docs)))

    return docs, tags
