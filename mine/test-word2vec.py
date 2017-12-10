from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
import numpy as np
# from babi_rnn_vi import vectorize_sentence

vectors = KeyedVectors.load_word2vec_format('outputs/vi.vec')

# print(vectors.wv)
print(vectors.wv['hôm_nay'])
# print(vectorize_sentence(['trời', 'hôm_nay', 'thế_nào', '?'], vectors))
