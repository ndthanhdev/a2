import numpy as np

from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


from tools import tokenize

'''
This model classification question
'''


def parse_question(lines, document_id):
	'''Parse question provided in babi format
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

def get_question(number_of_document):
	'''Get all question from 1_train.txt to number_of_document_train.txt
	'''
    data =[]
    for i in range(number_of_document):
        data.append(parse_question(open('{}_train.txt'.format(i), encoding='utf-8'),i))
	return data

def vectorize_(data, word2idx,number_of_document,question_maxlen):
    xs=[]
    ys=[]
    for question,document_id in data:
        x = [word2idx[w] for w in question]
        y = np.zeros(number_of_document)
        y[document_id]=1
        xs.append(x)
        ys.append(y)
    return pad_sequences(xs,maxlen=question_maxlen), np.array(ys)

if __name__ == '__main__':
