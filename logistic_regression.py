import numpy as np


class LogitRegression:
    def __init__(self, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.epochs):
            hypothesis = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(hypothesis)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        predicted_percent = self._sigmoid(np.dot(X, self.weights) + self.bias)
        predicted_class = [1 if i >= 0.5 else 0 for i in predicted_percent]
        return predicted_class

    def evaluate(self, ground_truth, prediction):
        acc = np.sum(ground_truth == prediction) / len(ground_truth)
        return acc

# TEST


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import linear_model

    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

    logit_reg_fc = LogitRegression(lr=0.01, epochs=1000)
    logit_reg_fc.fit(X_train, y_train)
    y_val_predict = logit_reg_fc.predict(X_val)
    acc = logit_reg_fc.evaluate(y_val, y_val_predict)
    print("Logit Reg from scratch accuracy: ", acc)

    logit_reg = linear_model.LogisticRegression()
    logit_reg.fit(X_train, y_train)
    y_val_predict = logit_reg.predict(X_val)
    acc = accuracy_score(y_val, y_val_predict)
    print('Logit Reg with sklearn accuracy:', acc)
