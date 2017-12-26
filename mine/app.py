# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from tools import tokenize
from ranking import ranking


def load_document(path):
    lines = open(path, encoding='utf-8').readlines()
    docs = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        line = line.strip()
        if '\t' not in line:
            docs.append(line)
    return docs


def load_stories(path):
    lines = open(path, encoding='utf-8').readlines()
    docs = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        line = line.strip()
        if '\t' not in line:
            docs.append(line)
    return docs


def load_documents(path, number_of_document):
    docs = []
    for i in range(number_of_document):
        docs.append(load_stories('{}/{}_train.txt'.format(path, i)))
    return docs


if __name__ == '__main__':

    number_of_document = 2
    ouput_path = "outputs/{}"
    model_path = ouput_path.format('{}_model.h5')
    model_context_path = ouput_path.format('{}_model_context.npy')
    data_path = 'data/babi/vi'

    print('Loading...')

    class_model = load_model(model_path.format('class'))
    # (word2idx,query_maxlen)
    class_model_context = np.load(
        model_context_path.format('class'))

    answer_models = []
    answer_model_contexts = []
    for i in range(number_of_document):
        answer_models.append(load_model(model_path.format(i)))
        answer_model_contexts.append(np.load(model_context_path.format(i)))

    docs = load_documents(data_path, number_of_document)

    # model = load_model(ouput_path.format('model.h5'))
    # [word2idx, story_maxlen, query_maxlen] = np.load(
    #     ouput_path.format('model_context.npy'))
    # idx2word = dict([(v, k) for k, v in word2idx.items()])

    # document = load_document(ouput_path.format('../data/babi/vi/_train.txt'))

    def toVec(word2idx, words):
        vectors = []
        for w in words:
            if w not in word2idx:
                print(w)
            else:
                vectors.append(word2idx[w])
        return vectors

    def toWord(idx, idx2word):
        return idx2word[idx]

    def predict(query):
        query = tokenize(query)
        print('query:', query)

        doc_id = np.argmax(class_model.predict(pad_sequences(
            [toVec(class_model_context[0], query)], class_model_context[1]), 32))
        return doc_id
        # ranked_documents = ranking(document, query)
        # # print('ranking:', ranked_documents)
        # corpus = max(ranked_documents.keys(), key=(
        #     lambda k: ranked_documents[k]))
        # corpus = tokenize(corpus)
        # print('corpus:', corpus)

        # input = [pad_sequences([toVec(corpus)], story_maxlen), pad_sequences(
        #     [toVec(query)], query_maxlen)]
        # output = model.predict(input, 32)
        # return toWord(np.argmax(output))

    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'bye':
            print('Bot: Tạm biệt')
            exit()
        elif '?' in temp:
            print('Bot: ', predict(temp))

    str(input('You: '))
