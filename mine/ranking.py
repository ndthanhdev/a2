from babi_rnn_vi import tokenize


def score(document, query):
    '''
        Score a document

        # Arguments
        document: str document
        query: list of token
    '''
    total = 0
    tokenized_doc = tokenize(document)
    for token in query:
        total += tokenized_doc.count(token)
    return total


def ranking(documents, query):
    '''
    Return ranked documents

    # Arguments
        documents: list of document
        query: list of token
    '''

    docs = set()
    for doc in documents:
        docs.add(doc)
    return dict((doc, score(doc, query)) for doc in docs)
