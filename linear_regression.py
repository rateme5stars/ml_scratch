from operations import Operation
import math
op = Operation()


class LinearRegressionFC:
    def __init__(self, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        for _ in range(self.epochs):
            y_hat = op.sum_1d(op.dot(X, self.weights), self.bias)

            dw = op.multiply(
                (1/n_samples),
                op.dot(op.transpose(X), op.minus(y_hat, y))
            )

            db = (1/n_samples) * sum(op.minus(y_hat, y))

            self.weights = op.minus(self.weights, op.multiply(self.lr, dw))
            self.bias -= self.lr * db

    def predict(self, X):
        prediction = op.sum_1d(op.dot(X, self.weights), self.bias)
        return prediction

    def evaluate(self, X_val, y_val):
        y_val_predicted = self.predict(X_val)
        mse = op.mean(op.pow_n(op.minus(y_val, y_val_predicted), 2))
        return math.sqrt(mse)

# TEST


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model
    import numpy as np

    X, y = datasets.make_regression(n_samples=1000, n_features=3, noise=30, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

    lrfc = LinearRegressionFC(lr=0.02, epochs=1000)
    lrfc.fit(X_train.tolist(), y_train.tolist())
    rmse_fc = lrfc.evaluate(X_val.tolist(), y_val.tolist())
    print("RMSE from scratch: ", rmse_fc)

    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_val_predict = lr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_predict))
    print('RMSE with sklearn:', rmse)
