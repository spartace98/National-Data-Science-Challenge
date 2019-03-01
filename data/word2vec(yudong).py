#!/usr/bin/env python3

import word2vec
import pickle

# Run word2phrase to group up similar words "Los Angeles" to "Los_Angeles"
word2vec.word2phrase('allwords.txt', 'allphrases.txt', verbose=True)

# Now actually train the word2vec model.
word2vec.word2vec('allphrases.txt', 'word2vec.bin', size=100, verbose=True)


# Pickle dump the model into a binary file
model = word2vec.load('word2vec.bin')

f = open("word2vec.pkl", "wb")
pickle.dump(model, f)
f.close()
