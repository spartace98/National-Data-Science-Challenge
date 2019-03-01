import pandas as pd
import numpy as np

training_data_filepath = "data/train.csv"
chunk_size = 1000

csv_headers = ["itemid", "title", "Category", "image_path"]

number_of_categories = 58


# Takes a string and converts it to an array of numbers
# Each word is a single number
# Each char is 2 digits
# Case insensitive (a = 10, b = 11, ... z = 25)
# Numbers are mapped like this (0 -> 50, 1 -> 51, ... 9 -> 59)
# Special symbols (./#-_!@...etc) ---> 99

def categoryToBinaryVector(category):
	binVec = [0] * number_of_categories
	binVec[category] = 1
	return binVec

def stringToNumberArray(title):
	numberArr = []
	title = title.lower()
	titleArr = title.split(" ")
	for word in titleArr:
		number = ""
		letterArr = list(word)
		for char in letterArr:
			if not char.isalpha():
				if char.isdigit():
					number += str(int(char) + 50)
				else:
					number += str(99)
				break

			number += str(ord(char) - 87)
		numberArr.append(float(number) / 10000.0)
	return numberArr



nn = CNN(30, 100, number_of_categories)
for chunk in pd.read_csv(training_data_filepath, chunksize=chunk_size):
	for i in range(10):
		titles = chunk[csv_headers[1]].apply(stringToNumberArray)
		expected_output = chunk[csv_headers[2]].apply(categoryToBinaryVector)
		
		combined = pd.concat([titles, expected_output], axis=1, join_axes=[titles.index])

		for i in range(700):
			nn.descend(combined['title'][i],  np.asarray([combined['Category'][i]], dtype=np.float32))

		correct = 0
		for i in range(300):
			correct += nn.check(combined['title'][i+700],  np.asarray([combined['Category'][i+700]], dtype=np.float32))

		print(correct / 300 * 100, "%")