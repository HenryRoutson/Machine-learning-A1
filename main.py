
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt
import csv
import math
import random
from collections import Counter


# import data
# https://diveintopython.org/learn/file-handling/csv

TEST_PATH = 'COMP30027_2024_asst1_data/winequality-test.csv'
TRAIN_PATH = 'COMP30027_2024_asst1_data/winequality-train.csv'

ATTRIBUTES = [
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
training_data = np.loadtxt(TRAIN_PATH, delimiter=",", dtype=float, skiprows=1)


def instance_attributes(instance) :
  assert(len(instance) == len(ATTRIBUTES))
  return instance[::-1]

def instance_label(instance) :
  assert(len(instance) == len(ATTRIBUTES))
  return instance[-1]

def run_knn_return_label(test_instance : list[float], n : int) -> any :

  most_similar_train_instances = sorted([
      (
        distance_euclidean(instance_attributes(train_instance), test_instance),
        random.randint(1, 1000000), # don't sort based on label
        instance_label(train_instance)
      ) 
      for train_instance in training_data ], 
    reverse=True
  )


  labels = [label for (distance, random, label) in most_similar_train_instances]

  labels_of_n_closest = labels[:n]
  most_common_label_of_n_closest = Counter(labels_of_n_closest).most_common()[0][0]

  print(labels_of_n_closest, most_common_label_of_n_closest)

  return most_common_label_of_n_closest

  


# test each instance in the test data
testing_data  = np.loadtxt(TEST_PATH,  delimiter=",", dtype=float, skiprows=1)
for test_instance in testing_data :
  run_knn_return_label(test_instance, 9)

