# -*- coding: utf-8 -*-
from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from tools import tokenize
from ranking import ranking_stories


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
    # (word2idx,story_maxlen,query_maxlen,idx2word)
    answer_model_contexts = []
    for i in range(number_of_document):
        answer_models.append(load_model(model_path.format(i)))
        [word2idx, story_maxlen, query_maxlen] = np.load(
            model_context_path.format(i))
        answer_model_contexts.append((word2idx, story_maxlen, query_maxlen, dict([
                                     (v, k) for k, v in word2idx.items()])))

    docs = load_documents(data_path, number_of_document)

    def toVec(word2idx, words):
        vectors = []
        for w in words:
            if w not in word2idx:
                print(w)
            else:
                vectors.append(word2idx[w])
        return vectors

    def toWord(idx2word, idx):
        return idx2word[idx]

    def predict(query):
        query = tokenize(query)
        print('\tquery:', query)

        doc_id = np.argmax(class_model.predict(pad_sequences(
            [toVec(class_model_context[0], query)], class_model_context[1]), 32))
        print('\tanswer_type:{}'.format(doc_id))

        ranked_stories = ranking_stories(query, docs[doc_id], doc_id)
        story = tokenize(ranked_stories[0])
        print('\tstory:', story)

        input = [pad_sequences([toVec(answer_model_contexts[doc_id][0], story)], answer_model_contexts[doc_id][1]), pad_sequences(
            [toVec(answer_model_contexts[doc_id][0], query)], answer_model_contexts[doc_id][2])]
        output = answer_models[doc_id].predict(input, 32)
        return toWord(answer_model_contexts[doc_id][3], np.argmax(output))

    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'bye':
            print('Bot: Tạm biệt')
            exit()
        elif '?' in temp:
            print('Bot: ', predict(temp))

    str(input('You: '))
