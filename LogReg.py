import numpy as np


class LogReg:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    @staticmethod
    def __add_intercept(x):
        intercept = np.ones((x.shape[0], 1))
        return np.concatenate((intercept, x), axis=1)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, x, y):
        if self.fit_intercept:
            X = self.__add_intercept(x)

        # weights initialization
        self.theta = np.zeros(x.shape[1])

        for i in range(self.num_iter):
            z = np.dot(x, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(x.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            z = np.dot(x, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)

            if self.verbose == True and i % 10000 == 0:
                print(f'loss: {loss} \t')

    def predict_prob(self, x):
        if self.fit_intercept:
            x = self.__add_intercept(x)

        return self.__sigmoid(np.dot(x, self.theta))

    def predict(self, x):
        return self.predict_prob(x).round()

    def accuracy(self, x, y, probathreshold=0.5):

        predicted_classes = (self.predict(x) >= probathreshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == y)
        return accuracy * 100
