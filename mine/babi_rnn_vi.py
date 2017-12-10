from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec

import vnTokenizer


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in vnTokenizer.tokenize(sent).split() if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        # line = line.decode('utf-8').strip()
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid.replace('\ufeff', ''))
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a.strip()))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)

    def flatten(data): return reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q,
            answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_sentence(text, word2vec):
    return np.average([word2vec.wv[w] for w in text], axis=0)


def vectorize_stories(data, word2vec, answer_idx):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = vectorize_sentence(story, word2vec)
        xq = vectorize_sentence(query, word2vec)
        # let's not forget that index 0 is reserved
        y = np.zeros(len(answer_idx) + 1)
        y[answer_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return np.array(xs), np.array(xqs), np.array(ys)


def main():

    RNN = recurrent.LSTM
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 40
    print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                               EMBED_HIDDEN_SIZE,
                                                               SENT_HIDDEN_SIZE,
                                                               QUERY_HIDDEN_SIZE))

    # try:
    #     path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
    # except:
    #     print('Error downloading dataset, please download it manually:\n'
    #           '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
    #           '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    # raise
    # tar = tarfile.open(path)
    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
    # challenge = 'data/babi/vi/qa1_single-supporting-fact_{}.txt'
    challenge = 'data/babi/vi/qa1_single-supporting-fact_{}.txt'
    # challenge = 'data/babi/vi/qa12_conjunction_{}.txt'
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
    # train = get_stories(tar.extractfile(challenge.format('train')))
    train = get_stories(open(challenge.format('train'), encoding='utf-8'))
    # test = get_stories(tar.extractfile(challenge.format('test')))
    test = get_stories(open(challenge.format('test'), encoding='utf-8'))

    answers = set()
    for story, q, answer in train + test:
        answers |= set([answer])
    answers = sorted(answers)

    word2vec = KeyedVectors.load_word2vec_format('outputs/word2vec.txt')
    vector_size = word2vec.vector_size

    # Reserve 0 for masking via pad_sequences
    answer_size = len(answers) + 1
    answer_idx = dict((c, i + 1) for i, c in enumerate(answers))

    x, xq, y = vectorize_stories(train, word2vec, answer_idx)
    tx, txq, ty = vectorize_stories(test, word2vec, answer_idx)

    print('vocab = {}'.format(answers))
    print('x.shape = {}'.format(x.shape))
    print('xq.shape = {}'.format(xq.shape))
    print('y.shape = {}'.format(y.shape))
    print('vector_size = {}'.format(vector_size))

    print('Build model...')

    sentence = layers.Input(shape=(vector_size,), dtype='int32')
    encoded_sentence = layers.Embedding(
        answer_size, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

    question = layers.Input(shape=(vector_size,), dtype='int32')
    encoded_question = layers.Embedding(
        answer_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(vector_size)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(answer_size, activation='softmax')(merged)

    model = Model([sentence, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training')
    model.fit([x, xq], y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.05)
    loss, acc = model.evaluate([tx, txq], ty,
                               batch_size=BATCH_SIZE)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    print('Saving model')
    model.save('outputs/babi.h5')
    np.save('outputs/model_context.npy', [answer_idx])


if __name__ == '__main__':
    main()
