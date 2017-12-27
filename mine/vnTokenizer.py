from py4j.java_gateway import JavaGateway,  GatewayParameters

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=23333))
tokenizer = gateway.entry_point.getTokenizer()


def tokenize(text, turn_on_sentence_detection=False):
    '''
        Tokenizing Vietnamese document
        >>> tokenize('vnTokenizer được viết bởi thầy Lê Hồng Phương bằng Java. Chạy được trên Python do Thanh sử dụng Py4j.',turn_on_sentence_detection=False)
        'vnTokenizer được viết bởi thầy Lê_Hồng_Phương bằng Java . Chạy được trên Python do Thanh sử_dụng Py4j .'
        >>> tokenize('vnTokenizer được viết bởi thầy Lê Hồng Phương bằng Java. Chạy được trên Python do Thanh sử dụng Py4j.',turn_on_sentence_detection=True)
        ['vnTokenizer được viết bởi thầy Lê_Hồng_Phương bằng Java .', 'Chạy được trên Python do Thanh sử_dụng Py4j .']
    '''
    if turn_on_sentence_detection:
        tokenizer.turnOnSentenceDetection()
        return list(tokenizer.tokenize(text))
    else:
        tokenizer.turnOffSentenceDetection()
        return list(tokenizer.tokenize(text))[0]


def segment(sentence):
    return tokenizer.segment(sentence)
