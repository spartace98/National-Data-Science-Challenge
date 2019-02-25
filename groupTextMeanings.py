#!/usr/bin/env python3

import numpy as np
import csv

# categories = open("./data/categories.json")
data_train = open("./data/train.csv", "r", newline='')
output_dir = "./data/words/"

words_dict = dict()
parentAsso = dict()
mappingDict = {
    "beauty" : 0,
    "fashion": 1,
    "mobile" : 2
}

def getParent(imgpath):
    return mappingDict[imgpath.split("_")[0]]

reader = csv.DictReader(data_train)
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
        if word in words_dict:
            words_dict[word][categ] += 1
            parentAsso[word][getParent(entry["image_path"])] += 1
        else:
            wArr = np.zeros((58,), dtype=int)
            wArr[categ] = 1
            words_dict[word] = wArr

            wArr = np.zeros((3,), dtype=int)
            wArr[getParent(entry["image_path"])] = 1
            parentAsso[word] = wArr
data_train.close()

print("\n\nSaving to output npz File")

for word in words_dict:
    # outFile = open("{}/{}.npz".format(output_dir, word), "wb")
    # print(word)
    # print(words_dict[word])
    # print(parentAsso[word])
    # print("")
    np.savez_compressed("{}/{}".format(output_dir, word), categories=words_dict[word], parents=parentAsso[word])
    # outFile.close()
