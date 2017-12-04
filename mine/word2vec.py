from subprocess import call
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from subprocess import os


def create(iter=5):
    home = os.getcwd()
    call(home + ".\\vnTokenizer\\vnTokenizer.bat -i ..\\data\\word2vec\\input.txt -o ..\\data\\word2vec\\output.txt")
    f = open(home + ".\\data\\word2vec\\output.txt", encoding="utf8")

    # đọc file đã được tokenize
    corpus = f.read().lower()

    # raw sentences is a list of sentences.
    raw_sentences = corpus.split('.')
    sentences = []
    for sentence in raw_sentences:
        sentences.append(sentence.split())

    print(sentences)
    model = Word2Vec(sentences, size=100, window=5,
                     min_count=1, workers=4, iter=iter)
    model.wv.save_word2vec_format("data/word2vec/vectors.txt")
    return model


def load():
    return KeyedVectors.load_word2vec_format("data/word2vec/vectors.txt")
