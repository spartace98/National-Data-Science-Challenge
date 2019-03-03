#!/usr/bin/env python3

import pandas as pd


'''
When a new instance of TrainingData is created, data is loaded, SHUFFLED and stored in
x_train, y_train variables

The general category, retrieved from the image filepath, is stored in the general_cat variable.
It is either "beauty", "mobile", or "fashion"

Call class method getTrainingData(category) to get all x_train, y_train, x_val and y_val for that category

ARGS: 
type - pass in "sentences" to get back a list of sentences for x_train
       pass in "words" to get back a list of every single word in the data for x_train

Example usage:
td = TrainingData("sentences")
x_train, y_train, x_val, y_val, output_size = td.getTrainingData("beauty")
'''

class TrainingData:
    def __init__(self, validation_percent=0.1, type="sentences"):
        self.data = pd.read_csv("data/train.csv", delimiter=",")
        # SHUFFLE
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.validation_percent = validation_percent

        self.x_train = []
        for i, sentence in enumerate(self.data["title"]):
            if type == "sentences":
                self.x_train.append(sentence)
            elif type == "words":
                for word in sentence:
                    self.x_train.append(word)
        
        self.y_train = []
        for i, category in enumerate(self.data["Category"]):
            self.y_train.append(category)

        self.general_cat = []
        self.general_cat_val = []
        for i, filepath in enumerate(self.data["image_path"]):
            self.general_cat.append(filepath.split("/")[0].split("_")[0])

    def getTrainingData(self, category = 0):
        indices = []
        for i, cat in enumerate(self.general_cat):
            if cat == category:
                indices.append(i)
        train = int(round(len(indices) * (1.0 - self.validation_percent)))

        x_train = []
        y_train = []
        x_val = []
        y_val = []
        
        for i, index in enumerate(indices):
            if i > train - 1:
                x_val.append(self.x_train[index])
                y_val.append(self.y_train[index])
                continue
            x_train.append(self.x_train[index])
            y_train.append(self.y_train[index])

        # CONVERT Y TO SET OF INDICES WITHIN THE CATEGORY ONLY
        minimum = min(s for s in y_train)
        y_train = [(y - minimum) for y in y_train]
        y_val = [(y - minimum) for y in y_val]

        output_size = len(set(y_val))

        return x_train, y_train, x_val, y_val, output_size



class TestData:
    def __init__(self):
        self.data = pd.read_csv("data/test.csv", delimiter=",")
        
        self.x_test = []
        for sentence in self.data["title"]:
            self.x_test.append(sentence)

        self.general_cat = []
        for filepath in self.data["image_path"]:
            self.general_cat.append(filepath.split("/")[0].split("_")[0])