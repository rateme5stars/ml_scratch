import numpy as np
from distance import Distance
from collections import Counter

class KnnNumpy:
    # Update more method in the future
    def __init__(self, k=5, method="brute_force"):
        self.k = k
        self.method = method

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _brute_force(self, x):
        # Calculate distance from input to every data point in training set
        distances = [Distance.euclidean(x, data_point) for data_point in X_train]

        # Indexing neighbors by distance and get k nearest neighbors indices
        knn_id = np.argsort(distances)[:self.k]

        # Get knn label
        knn_labels = [self.y_train[i] for i in knn_id]
        return knn_labels

    def predict(self, x):
        if self.method == "brute_force":
            knn_labels = self._brute_force(x)
            prediction = Counter(knn_labels).most_common(1)[0][0]
            return prediction


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    knn_fc = KnnNumpy(k=5)
    knn_fc.fit(X_train, y_train)
    predictions = np.array([knn_fc.predict(x) for x in X_test])
    acc = np.sum(predictions == y_test) / len(y_test)
    print("KNN from scratch accuracy: ", acc)

    knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print('KNN with sklearn accuracy:', acc)
