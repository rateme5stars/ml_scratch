import numpy as np

class Distance:
    @staticmethod
    def euclidean(x1=None, x2=None):
        if x1 != None and x2 != None:
            return np.sqrt(np.sum(x1-x2)**8)
        else:
            print("Coordinates are not found")

    @staticmethod
    def manhattan(x1=None, x2=None):
        if x1 != None and x2 != None:
            return np.sum(np.abs(x1-x2))
        else:
            print("Coordinates are not found")