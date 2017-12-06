from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from babi_rnn_vi import tokenize
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec


def main():
    print('Loading...')
    model = load_model('outputs/babi.h5')

    word_idx, story_maxlen, query_maxlen = np.load(
        'outputs/model_context.npy')
    idx_word = list(word_idx.keys())

    word2vec = KeyedVectors.load_word2vec_format('outputs/word2vec.txt')

    def mostSimilarity(externalVoca):
        # mostSimilarityIndex = np.argmax(
        #     [word2vec.wv.similarity(externalVoca, k) for k in word_idx if k in word2vec.wv.vocab])
        # print(word_idx, mostSimilarityIndex)
        return max([k for k in word_idx if k in word2vec.wv.vocab], key=lambda k: word2vec.wv.similarity(externalVoca, k))
        # return word_idx[mostSimilarityIndex - 1]

    def toVec(string):
        words = tokenize(string)
        vectors = []
        for w in words:
            if w not in word_idx:
                w = mostSimilarity(w)
            vectors.append(word_idx[w])
        return vectors

    def toWord(idx):
        return idx_word[idx - 1]

    def predict(corpus, query):
        input = [pad_sequences([toVec(corpus)], story_maxlen), pad_sequences(
            [toVec(query)], query_maxlen)]
        output = model.predict(input, 32)
        return toWord(np.argmax(output))

    corpus = ''
    print('Chào bạn!')
    while True:
        temp = str(input('You: '))
        if temp == 'exit':
            print('Bot: Tạm biệt')
            exit()
        elif temp == 'new':
            corpus = ''
            print('Bot: Quên hết rồi.')
        elif '?' in temp:
            print('Bot: ', predict(corpus, temp))
        else:
            corpus = corpus.strip() + ' ' + temp.strip()
            print('Bot: ờ')


if __name__ == '__main__':
    main()
