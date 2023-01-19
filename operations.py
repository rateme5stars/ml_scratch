import numbers
import numpy as np  # only use to calculate sigmoid function


class Operation:
    def sum_1d(self, x1, x2):
        # Boardcasting
        if isinstance(x1, numbers.Number):
            x1 = [x1] * len(x2)
        if (isinstance(x2, numbers.Number)):
            x2 = [x2] * len(x1)

        result = [0] * len(x1)
        for i in range(len(x1)):
            result[i] = x1[i] + x2[i]
        return result

    def multiply(self, x1, x2):
        # Boardcasting
        if isinstance(x1, numbers.Number):
            x1 = [x1] * len(x2)
        if isinstance(x2, numbers.Number):
            x2 = [x2] * len(x1)

        result = [0] * len(x1)
        for i in range(len(x1)):
            result[i] = round(x1[i] * x2[i], 8)
        return result

    def minus(self, x1, x2):
        # Boardcasting
        if isinstance(x1, numbers.Number):
            x1 = [x1] * len(x2)
        if (isinstance(x2, numbers.Number)):
            x2 = [x2] * len(x1)

        result = [0] * len(x1)
        for i in range(len(x1)):
            result[i] = x1[i] - x2[i]
        return result

    def divide(self, x1, x2):
        # Boardcasting
        if isinstance(x1, numbers.Number):
            x1 = [x1] * len(x2)
        if (isinstance(x2, numbers.Number)):
            x2 = [x2] * len(x1)

        result = [0] * len(x1)
        for i in range(len(x1)):
            result[i] = round(x1[i] / x2[i], 8)
        return result

    def sum_2d(self, x1, x2):
        result = [[0] * len(x1[0])] * len(x1)
        for i in range(len(x1)):
            result[i] = self.sum_1d(x1[i], x2[i])
        return result

    def transpose(self, x):
        if type(x[0]) is not list:
            return x
        result = [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]
        return result

    def dot(self, x1, x2):
        result = [0] * len(x1)
        x2_T = self.transpose(x2)
        for i in range(len(x1)):
            if type(x2_T[0]) is not list:
                result[i] = sum(self.multiply(x1[i], x2))
            else:
                tmp = []
                for j in range(len(x2_T)):
                    tmp.append(sum(self.multiply(x1[i], x2_T[j])))
                result[i] = tmp
        return result

    def mean(self, x, axis=None):
        if axis == 0:
            result = []
            x_T = self.transpose(x)
            for row in x_T:
                result.append(round(sum(row) / len(row), 8))
            return result
        else:
            return (sum(x) / len(x))

    def pow_n(self, x, n):
        return [num ** n for num in x]

    def sigmoid(self, x):
        x_np = np.array(x)
        sigmoid = 1 / (1 + np.exp(-x_np))
        return sigmoid.tolist()

    def argsort(self, x):
        return sorted(range(len(x)), key=x.__getitem__)

    def argmin(self, x):
        min_value = min(x)
        min_id = x.index(min_value)
        return min_id

    def argmax(self, x):
        max_value = max(x)
        max_id = x.index(max_value)
        return max_id
