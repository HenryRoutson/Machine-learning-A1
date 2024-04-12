
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt
import csv
import math
import random
from collections import Counter
import pandas as pd
from pandas import Series, DataFrame
from sklearn.metrics import confusion_matrix
import seaborn as sn


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
import math


def distance_euclidean(A : list[any], B : list[any]) -> float :
    assert(len(A) == len(B))
    return math.sqrt(sum([math.pow((t[0] - t[1]), 2) for t in zip(A, B) ]))

print(distance_euclidean([60], [42]))
assert(distance_euclidean([60], [42]) == 18.0)

def instance_label(instance) :
  assert(len(instance) == len(ATTRIBUTES))
  return instance[-1]

def instance_attributes(instance) :
  assert(len(instance) == len(ATTRIBUTES))
  return instance[::-1]


def attribute_distrib_summary(data) :

  print()
   
  for i, attrib in enumerate(ATTRIBUTES) :
    attrib_data = [row[i] for row in data]
    # print("range   ", attrib, max(attrib_data) -min(attrib_data))

    print("max     ", attrib, max(attrib_data))
    print("average ", attrib, sum(attrib_data) / len(attrib_data))
    print("min     ", attrib, min(attrib_data))
    print()

def data_label_distrib(data) :
  print(Counter([instance_label(row) for row in data]))


def serperateLabelsLowHigh(data) :
  return [ 
   list(filter(lambda row : row[-1] == 0, data)),
   list(filter(lambda row : row[-1] == 1, data))
  ]


def get_column(rows, index) :
  return [row[index] for row in rows]

def generate_all_scatterplots(data, data_name : str) :

  [low_rows, high_rows] = serperateLabelsLowHigh(data)

  for attrib_x_index, attrib_x in enumerate(ATTRIBUTES) :
    for attrib_y_index, attrib_y in enumerate(ATTRIBUTES) :

      fig, ax = plt.subplots()

      x_high = get_column(high_rows, attrib_x_index)
      x_low = get_column(low_rows, attrib_x_index)

      y_high = get_column(high_rows, attrib_y_index) 
      y_low =  get_column(low_rows, attrib_y_index) 

      # add in not high quality
      ax.scatter(x_low, y_low, c='red', alpha=0.3)
      
      # add in high quality
      ax.scatter(x_high, y_high, c="blue", alpha=0.3)
      
      plt.xlabel(attrib_x)
      plt.ylabel(attrib_y)

      plotTitle = attrib_x + "vs" + attrib_y
      plt.title(plotTitle)
      ax.legend()
      ax.grid(True)

      plt.savefig("graphs/"+data_name +"/"+plotTitle)

      plt.cla()
      plt.clf()







def generate_all_distributions(data, data_name : str) :

  [low_rows, high_rows] = serperateLabelsLowHigh(data)

  for attrib_index, attrib in enumerate(ATTRIBUTES) :

    high_values = get_column(high_rows, attrib_index)
    low_values = get_column(low_rows, attrib_index)

    # histogram looks weird
    plt.cla()
    plt.clf()

    plt.xlabel(attrib)
    plt.ylabel("frequency")

    plt.hist([high_values, low_values], bins=20, label=['high_values', 'low_values'], color=["red", "blue"])
    plt.savefig("hist_distributions/"+data_name +"/"+attrib)


    plt.cla()
    plt.clf()








def data_report(data) :
  print("DATA REPORT START ========================")
  print()
  data_label_distrib(data)
  print()
  attribute_distrib_summary(data)
  print()
  print("DATA REPORT END ========================")
  



# using loadtxt()
training_data = np.loadtxt(TRAIN_PATH, delimiter=",", dtype=float, skiprows=1)
data_report(training_data)
#generate_all_scatterplots(training_data, "train")
#generate_all_distributions(training_data, "train")

def run_knn_return_label(test_instance : list[float], n : int) -> any :

  most_similar_train_instances = sorted([
      (
        distance_euclidean(instance_attributes(train_instance), test_instance),
        instance_label(train_instance)
      ) 
      for train_instance in training_data ], 
    key = lambda x : x[0]
  )

  labels = [label for (distance, label) in most_similar_train_instances]

  labels_of_n_closest = labels[:n]
  most_common_label_of_n_closest = Counter(labels_of_n_closest).most_common()[0][0]

  #print(labels_of_n_closest, most_common_label_of_n_closest)

  return most_common_label_of_n_closest

  


def run_knn_on_test_data() :
  testing_data  = np.loadtxt(TEST_PATH,  delimiter=",", dtype=float, skiprows=1)

  # test each instance in the test data
  for test_instance in testing_data :
    predicted_label = run_knn_return_label(test_instance, 10)
    actual_label = instance_label(test_instance)
    print(predicted_label == actual_label, predicted_label, actual_label)



def check_accuracy() :
  testing_data  = np.loadtxt(TEST_PATH,  delimiter=",", dtype=float, skiprows=1)

  # test each instance in the test data
  predicted = [ run_knn_return_label(test_instance, 10) for test_instance in testing_data ]
  actual = [ instance_label(test_instance) for test_instance in testing_data ]
    
  conf = confusion_matrix(actual, predicted)


  accur = sum(conf[i][i] for i in range(len(conf))) / sum(sum(x) for x in conf)
  print("ACCURACY : " + str(accur))


  # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

  plt.cla()
  plt.clf()


  plt.figure(figsize = (10,7))

  # on the side is actual
  # on the bottom is predicted

  sn.heatmap(conf, annot=True, fmt=".0f")
  plt.savefig("conf")

  plt.cla()
  plt.clf()







check_accuracy()