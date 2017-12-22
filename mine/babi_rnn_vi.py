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

import vnTokenizer


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['bob', 'dropped', 'the', 'apple', '.', 'where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip().lower() for x in vnTokenizer.tokenize(sent).split() if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid.replace('\ufeff', ''))
        if nid == 1:
            story = []
        if '\t' in line:
            print(line)
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


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


if __name__ == '__main__':

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

    challenge = 'data/babi/vi/qa1_single-supporting-fact_{}.txt'

    train = get_stories(open(challenge.format('train'), encoding='utf-8'))

    vocab = set()
    for story, q, answer in train:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)

    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train )))
    query_maxlen = max(map(len, (x for _, x, _ in train )))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)

    print('vocab = {}'.format(vocab))
    print('x.shape = {}'.format(x.shape))
    print('xq.shape = {}'.format(xq.shape))
    print('y.shape = {}'.format(y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(
        story_maxlen, query_maxlen))

    print('Build model...')

    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    encoded_sentence = layers.Embedding(
        vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)

    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(
        vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)

    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(vocab_size, activation='softmax')(merged)

    model = Model([sentence, question], preds)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Training')
    model.fit([x, xq], y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.05)

    print('Saving model')
    model.save('outputs/model.h5')
    np.save('outputs/model_context.npy',
            [word_idx, story_maxlen, query_maxlen])
