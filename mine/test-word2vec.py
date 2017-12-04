import word2vec

vectors = word2vec.load()

print(vectors.most_similar(positive=["kim_đồng"]))