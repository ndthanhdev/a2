import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

text = 'Mary closed on closing night when she was on the mood to close'
sents = sent_tokenize(text)
print(sents)
words=[word_tokenize(sent) for sent in sents]
print(words)


from nltk.corpus import stopwords
from string import punctuation

customStopwords=set(stopwords.words('english') + list(punctuation))
wordsWOStopwords=[word for word in word_tokenize(text) if word not in customStopwords]
print(wordsWOStopwords)


from nltk.collocations import BigramCollocationFinder

bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(wordsWOStopwords)
print(sorted(finder.ngram_fd.items()))


from nltk.stem import LancasterStemmer

st = LancasterStemmer()
stemmedWords=[st.stem(word) for word in word_tokenize(text)]
print(stemmedWords)