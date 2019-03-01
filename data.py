#!/usr/bin/env python3

import pandas as pd


'''
When a new instance of TrainingData is created, data is loaded, SHUFFLED and stored in
x_train and y_train variables. 

The general category, retrieved from the image filepath, is stored in the general_cat variable.
It is either "beauty", "mobile", or "fashion"

ARGS: 
type - pass in "sentences" to get back a list of sentences for x_train
	   pass in "words" to get back a list of every single word in the data for x_train

Example usage:
td = TrainingData("sentences")
x_train = td.x_train
y_train = td.y_train

'''

class TrainingData:
	def __init__(self, type="sentences"):
		self.data = pd.read_csv("data/train.csv", delimiter=",")
		# SHUFFLE
		self.data = self.data.sample(frac=1).reset_index(drop=True)
		
		self.x_train = []
		for sentence in self.data["title"]:
			if type == "sentences":
				self.x_train.append(sentence)
			elif type == "words":
				for word in sentence:
					self.x_train.append(word)
		
		self.y_train = []
		for category in self.data["Category"]:
			self.y_train.append(category)

		self.general_cat = []
		for filepath in self.data["image_path"]:
			self.general_cat.append(filepath.split("/")[0].split("_")[0])


class TestData:
	def __init__(self):
		self.data = pd.read_csv("data/test.csv", delimiter=",")
		
		self.x_test = []
		for sentence in self.data["title"]:
			self.x_test.append(sentence)

		self.general_cat = []
		for filepath in self.data["image_path"]:
			self.general_cat.append(filepath.split("/")[0].split("_")[0])