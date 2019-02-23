import pandas as pd
import numpy as np

training_data_filepath = "data/train.csv"
chunk_size = 100

csv_headers = ["itemid", "title", "Category", "image_path"]


# Takes a string and converts it to an array of numbers
# Each word is a single number
# Each char is 2 digits
# Case insensitive (a = 10, b = 11, ... z = 25)
# Numbers are mapped like this (0 -> 50, 1 -> 51, ... 9 -> 59)
# Special symbols (./#-_!@...etc) ---> 99

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
		numberArr.append(int(number))
	return numberArr


for chunk in pd.read_csv(training_data_filepath, chunksize=chunk_size):
	titles = chunk[csv_headers[1]].apply(stringToNumberArray)
	expected_output = chunk[csv_headers[2]]


	break