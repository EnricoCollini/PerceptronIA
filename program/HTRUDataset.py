import numpy as np


#https://archive.ics.uci.edu/ml/datasets/HTRU2

htruDataSet = open('HTRUDataSet.txt')
h = htruDataSet.readlines()

for i in range(0, len(h)):
    h[i] = h[i].strip("\n")
    h[i] = h[i].split(",")

htruDataSetArray = np.zeros(shape=(len(h), len(h[0])))
for i in range(0, len(h)):
    for j in range(0, len(h[0])):
        htruDataSetArray[i][j] = h[i][j]

htruDataSetArrayShuffled = htruDataSetArray
htruDataSetArrayShuffled = np.random.permutation(htruDataSetArrayShuffled)

k1dataset1 = np.ones(shape=(int(len(htruDataSetArrayShuffled) / 4), len(htruDataSetArrayShuffled[0])))
k2dataset1 = np.ones(shape=(int(len(htruDataSetArrayShuffled) / 4), len(htruDataSetArrayShuffled[0])))
k3dataset1 = np.ones(shape=(int(len(htruDataSetArrayShuffled) / 4), len(htruDataSetArrayShuffled[0])))
k4dataset1 = np.ones(shape=(len(htruDataSetArrayShuffled) - len(k1dataset1) - len(k2dataset1) - len(k3dataset1), len(htruDataSetArrayShuffled[0])))
for i in range(0, len(htruDataSetArrayShuffled)):
    j = i % len(k1dataset1)
    if i < len(k1dataset1):
        k1dataset1[j] = htruDataSetArrayShuffled[i]
    elif i < len(k1dataset1) + len(k2dataset1):
        k2dataset1[j] = htruDataSetArrayShuffled[i]
    elif i < len(k1dataset1) + len(k2dataset1) + len(k3dataset1):
        k3dataset1[j] = htruDataSetArrayShuffled[i]
    else:
        k4dataset1[j] = htruDataSetArrayShuffled[i]

def getHTRUDataSetArray():
    return htruDataSetArray


def getHTRUDataSetArrayShuffled():
    return htruDataSetArrayShuffled


def getK1():
    return k1dataset1


def getK2():
    return k2dataset1


def getK3():
    return k3dataset1


def getK4():
    return k4dataset1


def createInputs(dataset):
    inputs = np.zeros(shape=(len(dataset), len(dataset[0]) - 1))
    for i in range(0, len(dataset)):
        for j in range(0, len(dataset[0]) - 1):
            inputs[i][j] = dataset[i][j]
    return inputs


def createLables(dataset):
    lables = np.zeros(shape=(len(dataset), 1))
    for i in range(0, len(dataset)):
        if dataset[i][len(dataset[0]) - 1] == 0:
            lables[i] = -1
        else:
            lables[i] = dataset[i][len(dataset[0]) - 1]
    return lables
