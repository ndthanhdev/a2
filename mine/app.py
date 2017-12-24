# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from babi_rnn_vi import tokenize
from ranking import ranking


def loadDocuments(path):
    lines = open(path, encoding='utf-8').readlines()
    docs=[]
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        line = line.strip()
        if '\t' not in line:
            docs.append(line)
    return docs

if __name__ == '__main__':

    source = "outputs/{}"

    print('Loading...')
    model = load_model(source.format('model.h5'))
    [word2idx, story_maxlen, query_maxlen] = np.load(
        source.format('model_context.npy'))
    idx2word = dict([(v, k) for k, v in word2idx.items()])

    documents = loadDocuments(source.format('../data/babi/vi/_train.txt'))

    def toVec(words):
        vectors = []
        for w in words:
            if w not in word2idx:
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
        # print('ranking:', ranked_documents)
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
