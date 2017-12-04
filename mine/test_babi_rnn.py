from keras.models import Model, load_model
import numpy as np
import re
from pyvi.pyvi import ViTokenizer, ViPosTagger
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.co

TOKENIZE_REGEX = '([^\wÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ]+)?'


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split(TOKENIZE_REGEX, ViTokenizer.tokenize(sent)) if x.strip()]

# def tokenize(sent):
#     '''Return the tokens of a sentence including punctuation.

#     >>> tokenize('Bob dropped the apple. Where is the apple?')
#     ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#     '''
#     return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


model = load_model('outputs/babi.h5')
word_idx = np.load('outputs/word_idx.npy').item()
idx_word = list(word_idx.keys())

corpus = tokenize(
    'Khang đi đến phòng tắm.')
corpus = [word_idx[w] for w in corpus]
question = tokenize('Khang ở đâu?')
query = [word_idx[w] for w in question]

input = [pad_sequences([corpus], 55), pad_sequences([query], 4)]
output = model.predict(input, 32)
idx = np.argmax(output)
print(idx_word[idx - 1])
