from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from babi_rnn_vi import vectorize_sentence
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from babi_rnn_vi import tokenize


def main():
    print('Loading...')
    model = load_model('outputs/babi.h5')

    [answer_idx] = np.load('outputs/model_context.npy')
    idx_word = list(answer_idx.keys())

    # word2vec = KeyedVectors.load_word2vec_format('outputs/vi.vec')
    word2vec = KeyedVectors.load_word2vec_format('outputs/word2vec.vec')

    def toWord(idx):
        return idx_word[idx - 1]

    def predict(corpus, query):
        corpus = tokenize(corpus)
        query = tokenize(query)
        print('corpus', corpus)
        print('query', query)
        input = [np.array([vectorize_sentence(corpus, word2vec)]),
                 np.array([vectorize_sentence(query, word2vec)])]
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


if __name__ == '__main__':
    main()
