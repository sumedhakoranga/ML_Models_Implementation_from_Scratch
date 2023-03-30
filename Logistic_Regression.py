import numpy as np


class LogisticRegression:
    def __init__(self, alpha=0.001, tolerance=1e-5):
        self.alpha = alpha
        self.tolerance = tolerance
        self.m = None
        self.n = None
        self.w = None

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
        return (self._predict_prob(X) > 0.5).astype(int).reshape((100,))

    def _predict_prob(self, X):
        y_pred_prob = 1/(1+np.exp(-X@self.w.transpose())).reshape((100,))
        return y_pred_prob

    def predict_prob(self, X):
        biased = np.ones((self.n, 1))
        X = np.concatenate((biased, X), axis=1)
        y_pred_prob = self._predict_prob(X)
        return y_pred_prob

    def random_w(self):
        self.w = np.random.rand(1, self.m+1)

    def score(self, X, y):
        y_pred = self.predict(X)
        correct = (y_pred == y)
        accuracy = correct.sum()/len(y)  # accuracy=correct predictions / total predictions
        return accuracy

    def gradient(self, X, y, y_pred_prob):
        dw = ((X.transpose())@(y_pred_prob-y)).reshape((1, 3))
        return dw

    def gradient_descent(self, X, y):
        while True:
            y_pred_prob = self._predict_prob(X)
            dw = self.gradient(X, y, y_pred_prob)
            new_w = self.w-self.alpha*dw
            if ((new_w-self.w) < self.tolerance).all():
                break
            else:
                self.w = new_w
