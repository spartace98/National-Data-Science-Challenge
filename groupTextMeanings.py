#!/usr/bin/env python3

import numpy as np
import csv
import Levenshtein

class groupTextMeanings():
    def __init__(self):
        # categories = open("./data/categories.json")
        self.data_train = open("./data/train.csv", "r", newline='')
        self.output_dir = "./data/words/"
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

        self.parseText()
        self.aggregateValues()

    def getParent(self, imgpath):
        return self.mappingDict[imgpath.split("_")[0]]

    def parseText(self):
        reader = csv.DictReader(self.data_train)
        i = 0
        for entry in reader:
            i += 1
            if i > 5:
                break
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
                    wArr = np.zeros((58,1), dtype=int)
                    wArr[categ] = 1
                    self.words_dict[word] = wArr

                    wArr = np.zeros((3,1), dtype=int)
                    wArr[self.getParent(entry["image_path"])] = 1
                    self.parentAsso[word] = wArr
        self.data_train.close()
        print("\n")

    def aggregateValues(self):
        for word in self.words_dict:
            # catSum = np.dot(self.words_dict[word], self.normalizationVector["words_dict"])/np.sum(self.words_dict[word]) + 29
            # parSum = np.dot(self.parentAsso[word], self.normalizationVector["parentAsso"])/np.sum(self.parentAsso[word]) + 1
            parentAssocArray = [self.parentAsso[word][0]] * 17 + [self.parentAsso[word][1]] * 14 + [self.parentAsso[word][2]] * 27
            tempArray = np.multiply(self.words_dict[word], np.array(parentAssocArray))
            print(tempArray)
            self.aggSum[word] = np.dot(tempArray, self.normalizationVector)/np.sum(tempArray) + 29
            print("{:>4}: {}".format(self.aggSum[word], word))

    def valueOf(self, word):
        pass

groupTextMeanings()



# print("\n\nSaving to output npz File")

# for word in words_dict:
#     # outFile = open("{}/{}.npz".format(output_dir, word), "wb")
#     # print(word)
#     # print(words_dict[word])
#     # print(parentAsso[word])
#     # print("")
#     np.savez_compressed("{}/{}".format(output_dir, word), categories=words_dict[word], parents=parentAsso[word])
#     # outFile.close()
