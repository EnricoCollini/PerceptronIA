import numpy as np

#bankDataSet
bankDataSet = open('BankDataSet.txt', 'r')
d = bankDataSet.readlines()


# Dalle righe del file testo ottengo una lista di tuple
for i in range(0, len(d)):
    d[i] = d[i].strip("\n")
    d[i] = d[i].split(",")


#Creo un array del bankDataSet
bankDataSetArray = np.zeros(shape=(len(d), len(d[0])))
for i in range(0, len(d)):
    for j in range(0, len(d[0])):
        bankDataSetArray[i][j] = d[i][j]





#Creo un array randomizzato del bankDataSet
bankDataSetArrayShuffled = bankDataSetArray
bankDataSetArrayShuffled = np.random.permutation(bankDataSetArrayShuffled)




#Genero 4 gruppi di partizionamento del bankDataSetShuffled per effettuare successivamente la k-cross-convalidation
k1dataset1 = np.ones(shape=(int(len(bankDataSetArrayShuffled) / 4), len(bankDataSetArrayShuffled[0])))
k2dataset1 = np.ones(shape=(int(len(bankDataSetArrayShuffled) / 4), len(bankDataSetArrayShuffled[0])))
k3dataset1 = np.ones(shape=(int(len(bankDataSetArrayShuffled) / 4), len(bankDataSetArrayShuffled[0])))
k4dataset1 = np.ones(shape=(len(bankDataSetArrayShuffled) - len(k1dataset1) - len(k2dataset1) - len(k3dataset1), len(bankDataSetArrayShuffled[0])))
for i in range(0, len(bankDataSetArrayShuffled)):
    j = i % len(k1dataset1)
    if i < len(k1dataset1):
        k1dataset1[j] = bankDataSetArrayShuffled[i]
    elif i < len(k1dataset1) + len(k2dataset1):
        k2dataset1[j] = bankDataSetArrayShuffled[i]
    elif i < len(k1dataset1) + len(k2dataset1) + len(k3dataset1):
        k3dataset1[j] = bankDataSetArrayShuffled[i]
    else:
        k4dataset1[j] = bankDataSetArrayShuffled[i]




#definisco delle funzioni per accedere ai dati del Dataset
def getBankDataSetArray():
    return bankDataSetArray


def getBankDataSetArrayShuffled():
    return bankDataSetArrayShuffled


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

bankDataSet.close()