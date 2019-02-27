#!/usr/bin/env python3

import numpy as np
import csv
import Levenshtein
import pickle
import os
import math

class groupTextMeanings():
    def __init__(self):
        if os.path.isfile("data/encoder.pkl"):
            f = open("data/encoder.pkl", "rb")
            self.aggSum, self.dbWordCount = pickle.load(f)
            f.close()
            return
        # categories = open("./data/categories.json")
        self.data_train = open("./data/train.csv", 'r' , newline='')
        self.output_dir = "./data/words/"
        # self.outputFile = open("{}/out.txt".format(self.output_dir), 'w')
        self.categoryNo = 58
        # self.normalizationVector = {
        #     "words_dict" : np.array(range(-29, 29)),    # -29 to 28
        #     "parentAsso" : np.array(range(-1, 2))       # -1  to 1
        # }
        self.normalizationVector = np.array(range(-29, 29)),    # -29 to 28

        self.words_dict = dict()
        self.parentAsso = dict()
        self.mappingDict = {
            "beauty" : 0,
            "fashion": 1,
            "mobile" : 2
        }

        self.aggSum  = dict()
        self.dbWordCount = 0


        self.parseText()
        self.aggregateValues()

    def getParent(self, imgpath):
        return self.mappingDict[imgpath.split("_")[0]]

    def parseText(self):
        reader = csv.DictReader(self.data_train)
        i = 0
        for entry in reader:
            i += 1
            # if i > 5:
            #     break
            # Print Progress
            print("Entry [{:>6}/666615] == Cat {:>2}: {}".format(i, entry["Category"], entry["title"]), end="\r")
            words = entry["title"].split(" ")
            categ = int(entry["Category"])

            # Process word
            for word in words:
                if word in self.words_dict:
                    self.words_dict[word][categ] += 1
                    self.parentAsso[word][self.getParent(entry["image_path"])] += 1
                else:
                    wArr = np.zeros(58)
                    wArr[categ] = 1
                    self.words_dict[word] = wArr

                    wArr = np.zeros(3)
                    wArr[self.getParent(entry["image_path"])] = 1
                    self.parentAsso[word] = wArr
        self.data_train.close()
        self.dbWordCount = len(self.words_dict)
        print("\n")

    def aggregateValues(self):
        print("Aggregating Values...", end="\r")
        for word in self.words_dict:
            # catSum = np.dot(self.words_dict[word], self.normalizationVector["words_dict"])/np.sum(self.words_dict[word]) + 29
            # parSum = np.dot(self.parentAsso[word], self.normalizationVector["parentAsso"])/np.sum(self.parentAsso[word]) + 1
            parentAssocArray = [self.parentAsso[word][0]] * 17 + [self.parentAsso[word][1]] * 14 + [self.parentAsso[word][2]] * 27
            tempArray = np.multiply(self.words_dict[word], np.array(parentAssocArray))
            # print(parentAssocArray)
            # print(tempArray)
            # print(self.normalizationVector)
            # aggSum = np.dot(self.normalizationVector, tempArray)/np.sum(tempArray) + 29
            self.aggSum[word] = tempArray / np.max(tempArray)
            # self.outputFile.write("{:>4}: {}\n".format(self.aggSum[word], word))
        
        # Pickle aggSum and dbWordCount
        f = open('data/encoder.pkl', 'wb')
        pickle.dump([self.aggSum, self.dbWordCount], f)
        f.close()

        print("Aggregating Values...done")

    def scurve(self, x):
        k = 0.99
        if x <= 0.5:
            return (k * 2 * x - 2 * x) / (4 * k * x - k - 1) * 0.5
        else:
            return 0.5 * ((-k) * 2 * (x - 0.5) - (2 * (x - 0.5))) / (2 * (-k) * 2 * (x - 0.5) + k - 1) + 0.5

    def valueOf(self, _word):
        if _word in self.aggSum:
            # Stop if word already exists
            return self.aggSum[_word]
        else:
            prev = 0
            for i, word in enumerate(self.aggSum):
                # https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html#Levenshtein-jaro
                jaro = Levenshtein.jaro(_word, word)
                if jaro > prev:
                    nArr = self.aggSum[word]
                    prev = jaro
            return nArr

#a = groupTextMeanings()
#print("End Calc")
#print(a.valueOf("cream"))
#print(a.valueOf("creamy"))
# Should give similar to one before



# print("\n\nSaving to output npz File")

# for word in words_dict:
#     # outFile = open("{}/{}.npz".format(output_dir, word), "wb")
#     # print(word)
#     # print(words_dict[word])
#     # print(parentAsso[word])
#     # print("")
#     np.savez_compressed("{}/{}".format(output_dir, word), categories=words_dict[word], parents=parentAsso[word])
#     # outFile.close()
