from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
from os import listdir
from os.path import isfile, join
from functools import reduce
import vnTokenizer


def create(iter=1000):
    path = 'data/word2vec'

    docLabels = []
    docLabels = [f for f in listdir(path) if f.endswith('.txt')]

    data = [open(join(path, doc), encoding='utf8').read() for doc in docLabels]
    data = '. '.join(data)

    sentences = []
    for sentence in vnTokenizer.tokenize(data, True):
        sentences.append(sentence.lower().split())

    model = FastText(sentences, size=100, window=5,
                     min_count=1, workers=4, iter=iter)
    model.wv.save_word2vec_format('outputs/word2vec.vec')

def main():
    create()


if __name__ == '__main__':
    main()
