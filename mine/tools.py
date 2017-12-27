import vnTokenizer


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['bob', 'dropped', 'the', 'apple', '.', 'where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip().lower() for x in vnTokenizer.tokenize(sent).split() if x.strip()]