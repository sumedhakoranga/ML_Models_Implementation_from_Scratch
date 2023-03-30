from cmath import sqrt
import numpy as np


class KNeighborsClassifier:
    def __init__(self, k=5, p=2):
        self.m = None
        self.n = None
        self.k = k
        self.p = p

    def fit(self, X, y):
        self.m = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y

    def _kneighbors(self, distance_):
        indices = np.argsort(distance_)
        k__neighbor = self.y[indices]
        return k__neighbor

    def predict(self, X):
        predictions = []
        for i in range(0, X.shape[0]):  # range is 0 to n_test
            sub_ = self.X-X[i]
            power_ = np.power(sub_, self.p)
            sum_ = np.sum(power_, axis=1)
            distance_ = np.sqrt(sum_)
            k_neighbor = self._kneighbors(distance_)
            value = self.majority_value(k_neighbor)
            predictions.append(value)
        return predictions

    # finding majority element in an array: we will use python Dictionary
    # initialize dictioary and count variable
    def majority_value(self, k_neighbor):
        d = {}
        count = 1
        # Traverse k_neighbor ,
        for i in k_neighbor:
            if i not in d:
                # if element is not in dictionary, add element to it.
                d[i] = count
            else:
                d[i] += count  # if it is in dictionary, update count of the element
            value = max(d, key=d.get)
            if d[i] >= (len(k_neighbor)//2):
                return value

    def score(self, X, y):
        # finding mean accuracy of preducted neighbors
        # we need to find mean of distance/average of distance
        y_pred = self.predict(X)
        correct = (y_pred == y)
        accuracy = correct.sum()/len(y)
        return accuracy
