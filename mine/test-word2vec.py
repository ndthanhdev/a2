from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

vectors = KeyedVectors.load_word2vec_format('outputs/word2vec.txt')

print(vectors.wv.most_similar('anh'))