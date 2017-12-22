# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from babi_rnn_vi import tokenize
from ranking import ranking


def loadDocuments(path):
    return open(path, encoding='utf-8').readlines()


if __name__ == '__main__':

    kind = "outputs/t1/{}"

    print('Loading...')
    model = load_model(kind.format('model.h5'))
    [word2idx, story_maxlen, query_maxlen] = np.load(
        kind.format('model_context.npy'))
    idx2word = dict([(v, k) for k, v in word2idx.items()])

    documents = loadDocuments(kind.format('documents.txt'))

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

    def predict(query):
        query = tokenize(query)
        print('query:', query)

        ranked_documents = ranking(documents, query)
        print('ranking:', ranked_documents)
        corpus = max(ranked_documents.keys(), key=(
            lambda k: ranked_documents[k]))
        corpus = tokenize(corpus)
        print('corpus:', corpus)

        input = [pad_sequences([toVec(corpus)], story_maxlen), pad_sequences(
            [toVec(query)], query_maxlen)]
        output = model.predict(input, 32)
        return toWord(np.argmax(output))

    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'bye':
            print('Bot: Tạm biệt')
            exit()
        elif '?' in temp:
            print('Bot: ', predict(temp))
