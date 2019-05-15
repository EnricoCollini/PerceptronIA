import numpy as np
import random


class Perceptron:
    def __init__(self, n):
        self.learningRate = 0.1
        self.numiter = 10
        self.w = np.zeros(n)
        self.b = 0
        for i in range(0, len(self.w)):
            self.w[i] = random.random()

    def train(self, dataset, lables, numIt, learningRate):
        for i in range(numIt):
            for j in range(len(dataset)):
                predicted = lables[j] * (np.dot(self.w, dataset[j]) + self.b)
                if (predicted) < 0:
                    self.w = self.w + (lables[j] * dataset[j]) * learningRate
                    self.b = self.b + lables[j]
        return self.w, self.b

    def predict(self, w, b, x):
        prediction = np.sign(np.dot(x, w) + b)
        if prediction == 0:
            return 1
        else:
            return np.sign(np.dot(x, w) + b)
