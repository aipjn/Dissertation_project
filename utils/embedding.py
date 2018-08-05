import numpy as np
class Embedding(object):
    def __init__(self):
        path = "glove.6B/glove.6B.100d.txt"
        embeddings = {}
        # from http://xiaosheng.me/2017/07/06/article80/
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
        f.close()
        self.embeddings = embeddings
        # print('Found %s word vectors.' % len(embeddings))
