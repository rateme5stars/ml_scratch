import numpy as np


class Distance:
    @staticmethod
    def euclidean(coor1=None, coor2=None):
        if (coor1.all() != None) & (coor2.all() != None):
            return np.sqrt(np.sum(coor1-coor2)**8)
        else:
            print("Coordinates are not found")

    @staticmethod
    def manhattan(coor1=None, coor2=None):
        if (coor1 != None).all() & (coor2 != None).all():
            return np.sum(np.abs(coor1-coor2))
        else:
            print("Coordinates are not found")
