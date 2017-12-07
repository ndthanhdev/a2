from py4j.java_gateway import JavaGateway,  GatewayParameters

gateway = JavaGateway(gateway_parameters=GatewayParameters(port=23333))
tokenizer = gateway.entry_point.getTokenizer()

def tokenize(text, turn_on_sentence_detection=False):
    if turn_on_sentence_detection:
        tokenizer.turnOnSentenceDetection()
        return list(tokenizer.tokenize(text))
    else:
        tokenizer.turnOffSentenceDetection()
        return list(tokenizer.tokenize(text))[0]


def segment(sentence):
    return tokenizer.segment(sentence)
