import numpy as np


class Perceptron:
    def __init__(self, alpha):
        self.w = np.zeros(2)
        self.alpha = alpha

    def train_wikipedia(self, x, y):
        x = Perceptron._add_column(x, 1)
        m = x.shape[0]
        features = x.shape[1]

        self.w = np.zeros(features)

        for i in range(m):
            prediction = np.dot(self.w, x.transpose())
            self.w = self.w + np.dot(y - prediction, x)
        return True

    def train(self, x, y):
        x = Perceptron._add_column(x, 1)
        m = x.shape[0]
        features = x.shape[1]

        self.w = np.zeros(features)

        found = True
        while found:
            found = False
            for i in range(m):
                prediction = np.dot(self.w, x[i])
                if prediction > 0:
                    prediction = 1
                else:
                    prediction = -1

                if prediction * y[i] < 0:
                    self.w = self.w + self.alpha * y[i] * x[i]
                    found = True

    def predict(self, x):
        x = Perceptron._add_element(x, 1)

        prediction = np.dot(self.w, x.transpose())
        if prediction > 0:
            return 1
        else:
            return 0

    @staticmethod
    def _add_column(a, n):
        ret = n * np.ones([a.shape[0], a.shape[1] + 1])
        for i in range(a.shape[0]):
            for j in range(1, ret.shape[1]):
                ret[i][j] = a[i][j - 1]

        return ret

    @staticmethod
    def _add_element(a, n):
        ret = n * np.ones([a.shape[0] + 1])
        for i in range(1, ret.shape[0]):
            ret[i] = a[i - 1]
        return ret