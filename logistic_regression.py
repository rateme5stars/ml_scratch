import numpy as np
from operations import Operation
op = Operation()


class LogitRegressionFC:
    def __init__(self, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        for _ in range(self.epochs):
            hypothesis = op.sum_1d(op.dot(X, self.weights), self.bias)
            y_hat = op.sigmoid(hypothesis)

            dw = op.multiply(
                (1/n_samples),
                op.dot(op.transpose(X), op.minus(y_hat, y))
            )
            db = (1/n_samples) * sum(op.minus(y_hat, y))

            self.weights = op.minus(self.weights, op.multiply(self.lr, dw))
            self.bias -= self.lr * db

    def predict(self, X):
        predicted_percent = op.sigmoid(op.sum_1d(np.dot(X, self.weights), self.bias))
        predicted_class = [1 if i >= 0.5 else 0 for i in predicted_percent]
        return predicted_class

    def evaluate(self, ground_truth, prediction):
        correct_predictions = 0
        for i in range(len(ground_truth)):
            if ground_truth[i] == prediction[i]:
                correct_predictions += 1
        acc = correct_predictions / len(ground_truth)
        return acc

# TEST


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn import linear_model
    import warnings
    warnings.filterwarnings("ignore")  # in order to ignore warning when print result

    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

    logit_reg_fc = LogitRegressionFC(lr=0.01, epochs=1000)
    logit_reg_fc.fit(X_train.tolist(), y_train.tolist())
    y_val_predict = logit_reg_fc.predict(X_val.tolist())
    acc = logit_reg_fc.evaluate(y_val, y_val_predict)
    print("Logit Reg from scratch accuracy: ", acc)

    logit_reg = linear_model.LogisticRegression()
    logit_reg.fit(X_train, y_train)
    y_val_predict = logit_reg.predict(X_val)
    acc = accuracy_score(y_val, y_val_predict)
    print('Logit Reg with sklearn accuracy:', acc)
