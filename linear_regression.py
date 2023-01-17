import numpy as np

class LinearRegressionNumpy:
    def __init__(self, lr=0.001, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        return prediction

    def evaluate(self, X_val, y_val):
        y_val_predicted = self.predict(X_val)
        mse = np.mean((y_val - y_val_predicted) ** 2)
        return np.sqrt(mse)

# TEST

if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn import linear_model

    X, y = datasets.make_regression(n_samples=1000, n_features=3, noise=30, random_state=3)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

    lrnp = LinearRegressionNumpy(lr=0.01, epochs=1000)
    lrnp.fit(X_train, y_train)
    rmse_np = lrnp.evaluate(X_val, y_val)
    print("RMSE with Numpy: ", rmse_np)

    lr = linear_model.LinearRegression()
    lr.fit(X_train, y_train)
    y_val_predict = lr.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_predict))
    print('RMSE:', rmse)
