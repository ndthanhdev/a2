from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from os import listdir
from os.path import isfile, join
from functools import reduce
import vnTokenizer


def create(iter=100):
    corpuses = []
    path = 'data/word2vec'
    files = [join(path, f)
             for f in listdir(path) if isfile(join(path, f))]
    corpuses = [open(fl, encoding='utf8').read() for fl in files]
    sentences = []
    for corpus in corpuses:
        for sentence in vnTokenizer.tokenize(corpus, True):
            sentences.append(sentence.split())
    model = Word2Vec(sentences, size=300, window=5,
                     min_count=1, workers=4, iter=iter)
    model.wv.save_word2vec_format('outputs/word2vec.txt')


def main():
    create()


if __name__ == '__main__':
    main()
