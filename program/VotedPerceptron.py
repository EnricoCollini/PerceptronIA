import numpy as np
import random


class VotedPerceptron:
    def __init__(self, n):
        self.learningRate = 0.1
        self.numiter = 10
        self.w = np.zeros(n)
        self.b = 1
        self.perceptrons = []
        self.times = []
        self.bias = []

    def set(self):
        self.perceptrons = []
        self.times = []
        self.bias = []

    def train(self, dataset, lables, numIt, learningRate):
        t = 1
        for i in range(numIt):
            for j in range(len(dataset)):
                predicted = lables[j] * (np.dot(self.w, dataset[j]) + self.b)
                if (predicted) < 0:
                    self.w = self.w + (lables[j] * dataset[j]) * learningRate
                    self.perceptrons.append(self.w)
                    self.times.append(t)
                    self.bias.append(self.b)
                    t = 1
                    self.b = self.b + lables[j]
                else:
                    t = t + 1
        return self.w, self.b

    def predict(self, b, x):
        p = 0
        i = 0
        while(i<len(self.perceptrons)):
            w = self.perceptrons[i]
            t = self.times[i]
            b = self.bias[i]
            i = i+1
            prediction = np.sign(np.dot(x, w)+b)
            p += t*prediction
            i = i + 1
        return np.sign(p)
