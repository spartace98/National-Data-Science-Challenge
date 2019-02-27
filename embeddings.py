from gensim.models import Word2Vec

f = open("data/allwords.txt", "r")

model = Word2Vec(f, size=100, window=5, min_count=1, workers=4)
model.save("data/allwords.model")