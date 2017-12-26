import numpy as np

from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model

from tools import tokenize

'''
This model classification question
'''


def parse_query(lines, document_id):
    '''
        Parse query provided in babi format
    '''
    data = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        if '\t' in line:
            print(line)
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            data.append((q, document_id))
    return data


def get_query(number_of_document):
    '''
        Get all query from 1_train.txt to number_of_document_train.txt
    '''
    data = []
    for i in range(number_of_document):
        data.extend(parse_query(
            open('data/babi/vi/{}_train.txt'.format(i), encoding='utf-8'), i))
    return data


def vectorize_query(data, word2idx, number_of_document, query_maxlen):
    xs = []
    ys = []
    for question, document_id in data:
        x = [word2idx[w] for w in question]
        y = np.zeros(number_of_document)
        y[document_id] = 1
        xs.append(x)
        ys.append(y)
    return pad_sequences(xs, maxlen=query_maxlen), np.array(ys)


if __name__ == '__main__':
    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 300
    BATCH_SIZE = 32
    EPOCHS = 100

    print('RNN / Embed = {}, {}'.format(RNN, EMBED_HIDDEN_SIZE))

    number_of_document = 2

    train = get_query(number_of_document)

    vocab = set()
    for q, doc in train:
        vocab |= set(q)
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word2idx = dict((c, i + 1) for i, c in enumerate(vocab))
    query_maxlen = max(map(len, (x for x, _ in train)))

    x, y = vectorize_query(train, word2idx, number_of_document, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('x.shape = {}'.format(x.shape))
    print('y.shape = {}'.format(y.shape))
    print('query_maxlen = {}'.format(query_maxlen))

    print('Build model...')

    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(
        vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    preds = layers.Dense(number_of_document,
                         activation='softmax')(encoded_question)

    model = Model(question, preds)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # plot model
    plot_model(model, to_file='class_rnn_model.png')

    print('Training')

    ch = 'y'
    while True:
        if ch == 'y':
            model.fit(x, y,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      validation_split=0.0)
        elif ch == 'n':
            break
        ch = str(input('Do you want continue train 100 Epochs?(y/n)')).strip()

    print('Saving model')
    model.save('outputs/class_model.h5')
    np.save('outputs/class_model_context.npy',
            [word2idx, query_maxlen])
