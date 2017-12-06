from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from pyvi.pyvi import ViTokenizer


def create(iter=100):
    corpus = open('data/word2vec/tenrieng.txt', encoding='utf8').read()
    sentences = []
    for sentence in ViTokenizer.tokenize(corpus).split('.'):
        sentences.append(sentence.split())
    model = Word2Vec(sentences, size=300, window=5,
                     min_count=1, workers=4, iter=iter)
    model.wv.save_word2vec_format('outputs/word2vec.txt')

def main():
    create()


if __name__ == '__main__':
    main()
