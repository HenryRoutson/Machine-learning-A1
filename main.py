
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt
import csv
import math


# import data
# https://diveintopython.org/learn/file-handling/csv

TEST_PATH = 'COMP30027_2024_asst1_data/winequality-test.csv'
TRAIN_PATH = 'COMP30027_2024_asst1_data/winequality-train.csv'

LABELS = [
    'fixedAcidity',
    'volatileAcidity',
    'citricAcid',
    'residualSugar',
    'chlorides',
    'freeSulfurDioxide',
    'totalSulfurDioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
    'quality'
    ]


# label_indexes
fixedAcidity = 0,
volatileAcidity = 1,
citricAcid = 2,
residualSugar = 3
chlorides = 4
freeSulfurDioxide = 5
totalSulfurDioxide = 6
density = 7
pH = 8
sulphates = 9 
alcohol = 10
quality = 11


        

# KNN

# create index constants




# 


def distance_euclidean(A : list[any], B : list[any]) -> float :
    assert(len(A) == len(B))
    return sum([(t[0] - t[1])**2 for t in zip(A, B) ])**0.5
    
assert(distance_euclidean([1], [2]) == 1.0)



 
# using loadtxt()
all_data = np.loadtxt(TRAIN_PATH, delimiter=",", dtype=float, skiprows=1)


for row in all_data :
    for row2 in all_data :
      print(distance_euclidean(row, row2))

