# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from babi_rnn_vi import tokenize

if __name__ == '__main__':

    kind = "outputs/t1_{}"

    print('Loading...')
    model = load_model(kind.format('model.h5'))
    [word2idx, story_maxlen, query_maxlen] = np.load(
       kind.format('model_context.npy'))
    idx2word = dict([(v, k) for k, v in word2idx.items()])
    # word2vec = KeyedVectors.load_word2vec_format('outputs/vi.vec')
    # word2vec = KeyedVectors.load_word2vec_format('outputs/word2vec.vec')

    def mostSimilarity(externalVoca):
        return max([k for k in word2idx if k in word2vec.wv.vocab], key=lambda k: word2vec.wv.similarity(externalVoca, k))

    def toVec(words):
        vectors = []
        for w in words:
            if w not in word2idx:
                # w = mostSimilarity(w)
                print(w)
            else:
                vectors.append(word2idx[w])
        return vectors

    def toWord(idx):
        return idx2word[idx]

    def predict(corpus, query):
        corpus = tokenize(corpus)
        query = tokenize(query)
        print('corpus:', corpus)
        print('query:', query)
        input = [pad_sequences([toVec(corpus)], story_maxlen), pad_sequences(
            [toVec(query)], query_maxlen)]
        output = model.predict(input, 32)
        return toWord(np.argmax(output))

    corpus = []
    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'exit':
            print('Bot: Tạm biệt')
            exit()
        elif temp == 'new':
            corpus = []
            print('Bot: Quên hết rồi.')
        elif '?' in temp:
            print('Bot: ', predict('. '.join(corpus), temp))
        else:
            corpus.append(temp.strip())
            print('Bot: ờ')
