import Perceptron
import VotedPerceptron
import HTRUDataset as Dataset
import numpy as np

# ottengo l'HTRUdataset
dataset = Dataset.getHTRUDataSetArrayShuffled()
perceptron = Perceptron.Perceptron(len(dataset[0]) - 1)
votedperceptron = VotedPerceptron.VotedPerceptron(len(dataset[0]) - 1)

k1dataset1 = Dataset.getK1()
k2dataset1 = Dataset.getK2()
k3dataset1 = Dataset.getK3()
k4dataset1 = Dataset.getK4()

k1dataset1inputs = Dataset.createInputs(k1dataset1)
k1dataset1lables = Dataset.createLables(k1dataset1)

k2dataset1inputs = Dataset.createInputs(k2dataset1)
k2dataset1lables = Dataset.createLables(k2dataset1)

k3dataset1inputs = Dataset.createInputs(k3dataset1)
k3dataset1lables = Dataset.createLables(k3dataset1)

k4dataset1inputs = Dataset.createInputs(k4dataset1)
k4dataset1lables = Dataset.createLables(k4dataset1)

# inizializzo i dati di testing per il perceptron
c1 = 0  # totale classe 1
c2 = 0  # totale classe 2
c1p = 0  # totale predetti classe 1
c2p = 0  # totale predetti classe 2
c1r1 = 0  # totale predetti classe 1 giusti
c1w1 = 0  # totale predetti classe 1 sbagliati
c2r2 = 0  # totale predetti classe 2 giusti
c2w2 = 0  # totale predetti calsse 2 sbagliati

# inizializzo i dati di testing per il voted perceptron
c1v = 0  # totale classe 1
c2v = 0  # totale classe 2
c1pv = 0  # totale predetti classe 1
c2pv = 0  # totale predetti classe 2
c1r1v = 0  # totale predetti classe 1 giusti
c1w1v = 0  # totale predetti classe 1 sbagliati
c2r2v = 0  # totale predetti classe 2 giusti
c2w2v = 0  # totale predetti calsse 2 sbagliati

# Training con KCrossConvalidation 1
perceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
perceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
perceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)

print("inizio training e testing 1")
votedperceptron.set()
votedperceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
votedperceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
votedperceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)

# testing sul gruppo 4 perceptron
realclassvalues = np.zeros(len(k4dataset1))
predicted = np.zeros(len(k4dataset1))
for i in range(0, len(k4dataset1)):
    realclassvalues[i] = k4dataset1lables[i]
    predicted[i] = perceptron.predict(perceptron.w, perceptron.b, k4dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1 = c1 + 1
    else:
        c2 = c2 + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1p = c1p + 1
        if realclassvalues[i] == -1:
            c1r1 = c1r1 + 1
        else:
            c1w1 = c1w1 + 1

    else:
        c2p = c2p + 1
        if realclassvalues[i] == 1:
            c2r2 = c2r2 + 1
        else:
            c2w2 = c2w2 + 1

# testing sul gruppo 4 votedperceptron
realclassvalues = np.zeros(len(k4dataset1))
predicted = np.zeros(len(k4dataset1))
for i in range(0, len(k4dataset1)):
    realclassvalues[i] = k4dataset1lables[i]
    predicted[i] = votedperceptron.predict(votedperceptron.b, k4dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1v = c1v + 1
    else:
        c2v = c2v + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1pv = c1pv + 1
        if realclassvalues[i] == -1:
            c1r1v = c1r1v + 1
        else:
            c1w1v = c1w1v + 1

    else:
        c2pv = c2pv + 1
        if realclassvalues[i] == 1:
            c2r2v = c2r2v + 1
        else:
            c2w2v = c2w2v + 1
print("training e testing 1 completato")

print("inizio training e testing 2")

# Training con KCrossConvalidation 2
perceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
perceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
perceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

votedperceptron.set()
votedperceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
votedperceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
votedperceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

# testing sul gruppo 3 perceptron
realclassvalues = np.zeros(len(k3dataset1))
predicted = np.zeros(len(k3dataset1))
for i in range(0, len(k3dataset1)):
    realclassvalues[i] = k3dataset1lables[i]
    predicted[i] = perceptron.predict(perceptron.w, perceptron.b, k3dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1 = c1 + 1
    else:
        c2 = c2 + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1p = c1p + 1
        if realclassvalues[i] == -1:
            c1r1 = c1r1 + 1
        else:
            c1w1 = c1w1 + 1

    else:
        c2p = c2p + 1
        if realclassvalues[i] == 1:
            c2r2 = c2r2 + 1
        else:
            c2w2 = c2w2 + 1

# testing sul gruppo 3 votedperceptron
realclassvalues = np.zeros(len(k3dataset1))
predicted = np.zeros(len(k3dataset1))
for i in range(0, len(k3dataset1)):
    realclassvalues[i] = k3dataset1lables[i]
    predicted[i] = votedperceptron.predict(votedperceptron.b, k3dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1v = c1v + 1
    else:
        c2v = c2v + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1pv = c1pv + 1
        if realclassvalues[i] == -1:
            c1r1v = c1r1v + 1
        else:
            c1w1v = c1w1v + 1

    else:
        c2pv = c2pv + 1
        if realclassvalues[i] == 1:
            c2r2v = c2r2v + 1
        else:
            c2w2v = c2w2v + 1

print("training e testing 2 completato")


print("inizio training e testing 3")
# Training con KCrossConvalidation 3
perceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
perceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)
perceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

votedperceptron.set()
votedperceptron.train(k1dataset1inputs, k1dataset1lables, 10, 0.1)
votedperceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)
votedperceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

# testing sul gruppo 2 perceptron
realclassvalues = np.zeros(len(k2dataset1))
predicted = np.zeros(len(k2dataset1))
for i in range(0, len(k2dataset1)):
    realclassvalues[i] = k2dataset1lables[i]
    predicted[i] = perceptron.predict(perceptron.w, perceptron.b, k2dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1 = c1 + 1
    else:
        c2 = c2 + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1p = c1p + 1
        if realclassvalues[i] == -1:
            c1r1 = c1r1 + 1
        else:
            c1w1 = c1w1 + 1

    else:
        c2p = c2p + 1
        if realclassvalues[i] == 1:
            c2r2 = c2r2 + 1
        else:
            c2w2 = c2w2 + 1

# testing sul gruppo 2 votedperceptron
realclassvalues = np.zeros(len(k2dataset1))
predicted = np.zeros(len(k2dataset1))
for i in range(0, len(k2dataset1)):
    realclassvalues[i] = k2dataset1lables[i]
    predicted[i] = votedperceptron.predict(votedperceptron.b, k2dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1v = c1v + 1
    else:
        c2v = c2v + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1pv = c1pv + 1
        if realclassvalues[i] == -1:
            c1r1v = c1r1v + 1
        else:
            c1w1v = c1w1v + 1

    else:
        c2pv = c2pv + 1
        if realclassvalues[i] == 1:
            c2r2v = c2r2v + 1
        else:
            c2w2v = c2w2v + 1

print("training e testing 3 completato")

print("inizio training e testing 4")
# Training con KCrossConvalidation 4
perceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
perceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)
perceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

votedperceptron.set()
votedperceptron.train(k2dataset1inputs, k2dataset1lables, 10, 0.1)
votedperceptron.train(k3dataset1inputs, k3dataset1lables, 10, 0.1)
votedperceptron.train(k4dataset1inputs, k4dataset1lables, 10, 0.1)

# testing sul gruppo 1 perceptron
realclassvalues = np.zeros(len(k1dataset1))
predicted = np.zeros(len(k1dataset1))
for i in range(0, len(k1dataset1)):
    realclassvalues[i] = k1dataset1lables[i]
    predicted[i] = perceptron.predict(perceptron.w, perceptron.b, k1dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1 = c1 + 1
    else:
        c2 = c2 + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1p = c1p + 1
        if realclassvalues[i] == -1:
            c1r1 = c1r1 + 1
        else:
            c1w1 = c1w1 + 1

    else:
        c2p = c2p + 1
        if realclassvalues[i] == 1:
            c2r2 = c2r2 + 1
        else:
            c2w2 = c2w2 + 1

# testing sul gruppo 1 votedperceptron
realclassvalues = np.zeros(len(k1dataset1))
predicted = np.zeros(len(k1dataset1))
for i in range(0, len(k1dataset1)):
    realclassvalues[i] = k1dataset1lables[i]
    predicted[i] = votedperceptron.predict(votedperceptron.b, k1dataset1inputs[i])

# getting the results:
for i in range(len(realclassvalues)):
    if realclassvalues[i] == -1:
        c1v = c1v + 1
    else:
        c2v = c2v + 1

for i in range(len(predicted)):
    if predicted[i] == -1:
        c1pv = c1pv + 1
        if realclassvalues[i] == -1:
            c1r1v = c1r1v + 1
        else:
            c1w1v = c1w1v + 1

    else:
        c2pv = c2pv + 1
        if realclassvalues[i] == 1:
            c2r2v = c2r2v + 1
        else:
            c2w2v = c2w2v + 1
print("training e testing 4 completato")

def getResults():
    r1 = c1r1, c2w2, c1
    r2 = c1w1, c2r2, c2
    r3 = c1p, c2p, (c1 + c2)
    acc = (c1r1 + c2r2) / (c1 + c2)
    return r1, r2, r3, acc


def getResultsVoted():
    r1 = c1r1v, c2w2v, c1v
    r2 = c1w1v, c2r2v, c2v
    r3 = c1pv, c2pv, (c1v + c2v)
    acc = (c1r1v + c2r2v) / (c1v + c2v)
    return r1, r2, r3, acc
