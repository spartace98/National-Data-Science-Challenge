#!/usr/bin/env python3

import word2vec
import pickle

# Run word2phrase to group up similar words "Los Angeles" to "Los_Angeles"
word2vec.word2phrase('data/allwords.txt', 'data/allphrases.txt', verbose=True)

# Now actually train the word2vec model.
word2vec.word2vec('data/allphrases.txt', 'data/word2vec.bin', size=100, verbose=True)


# Pickle dump the model into a binary file
model = word2vec.load('data/word2vec.bin')

f = open("data/word2vec.pkl", "wb")
pickle.dump(model, f)
f.close()
