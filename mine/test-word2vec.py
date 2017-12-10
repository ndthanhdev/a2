from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText
import numpy as np
# from babi_rnn_vi import vectorize_sentence

vectors = FastText.load_word2vec_format('outputs/vi.bin',binary=True)

print(vectors.wv)
# print(vectors.wv['bầu_trời'])
# print(vectorize_sentence(['trời', 'hôm_nay', 'thế_nào', '?'], vectors))
