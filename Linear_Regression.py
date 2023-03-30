import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.01, tolerance=1e-5):
        self.alpha = alpha
        self.tolerance = tolerance
        self.m = None
        self.n = None
        self.W = None

    def fit(self, X, y):
        self.m = X.shape[1]
        self.n = X.shape[0]
        biased = np.ones((self.n, 1))
        X = np.concatenate((biased, X), axis=1)

        self.random_w()
        self.gradient_descent(X, y)

    def predict(self, X):
        biased = np.ones((self.n, 1))
        X = np.concatenate((biased, X), axis=1)

        return self._predict(X)

    def _predict(self, X):
        return self.W@X.transpose()

    def score(self, X, y):
        y_pred = self.predict(X)
        biased = np.ones((self.n, 1))
        X = np.concatenate((biased, X), axis=1)
        r_squared = 1-((y_pred-y)@(y_pred-y).transpose())/(y@(y.transpose()))
        return r_squared[0][0]

    def random_w(self):
        self.W = np.random.rand(1, self.m+1)

    def gradient_descent(self, X, y):
        while True:
            dw = self.gradient(X, y)

            new_W = self.W - self.alpha*dw
            if ((new_W - self.W) < self.tolerance).all():
                break
            self.W = new_W

    def gradient(self, X, y):
        y_pred = self._predict(X)
        dw = ((X.transpose()@(y_pred-y).transpose())).transpose()/self.n
        return dw
