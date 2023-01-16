import numpy as np
from distance import Distance
from collections import Counter

class KnnFC:
    def __init__(self, k=5, method="brute_force"):
        self.k = k
        self.method = method
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _brute_force(self, x):
        distances = [Distance.euclidean(x, data_point) for data_point in X_train]
        return distances        
    
    def predict(self, x):
        if self.method == "brute_force":
            predicted_labels = [self._brute_force(feature) for feature in x]
            return np.array(predicted_labels)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    

                                                                                                           