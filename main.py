
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



with open(TEST_PATH, 'r') as file:
    reader = csv.reader(file)

    for A in reader:
        for B in reader : 
            print(scipy.spatial.distance.euclidean(A, B))


