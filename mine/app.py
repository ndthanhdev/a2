from keras.models import Model, load_model
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from babi_rnn_vi import tokenize


def main():
    print('Loading model')
    model = load_model('outputs/babi.h5')

    word_idx, story_maxlen, query_maxlen = np.load(
        'outputs/model_context.npy')
    idx_word = list(word_idx.keys())

    def toVec(string):
        words = tokenize(string)
        return [word_idx[w] for w in words]

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
            exit()
        elif temp == 'new':
            corpus = ''
            print('Bot: Quên hết rồi.')
        elif '?' in temp:
            print('Bot: ', corpus, '=>', predict(corpus, temp))
        else:
            corpus = corpus.strip() + ' ' + temp.strip()
            print('Bot: ờ')


if __name__ == '__main__':
    main()
