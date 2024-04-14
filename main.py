
import numpy as np
import sklearn
import scipy
import matplotlib.pyplot as plt
import random
from collections import Counter
import pandas as pd
from pandas import Series, DataFrame
from sklearn.metrics import confusion_matrix
import seaborn as sn
import random
import math
from typing import Callable



####################################################################################################################################
# Constants

TEST_PATH = 'COMP30027_2024_asst1_data/winequality-test.csv'
TRAIN_PATH = 'COMP30027_2024_asst1_data/winequality-train.csv'

TRAINING_DATA = np.loadtxt(TRAIN_PATH, delimiter=",", dtype=float, skiprows=1)

TESTING_DATA  = np.loadtxt(TEST_PATH,  delimiter=",", dtype=float, skiprows=1)


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


LOW_QUALITY = 0
HIGH_QUALITY = 1

label_t = int



####################################################################################################################################
# Functions

def distance_euclidean(A : list[float], B : list[float]) -> float :
    assert(len(A) == len(B))
    return math.sqrt(sum([math.pow((t[0] - t[1]), 2) for t in zip(A, B) ]))

assert(distance_euclidean([60], [42]) == 18.0)

def instance_label(instance) -> label_t :
  assert(len(instance) == len(ATTRIBUTES))
  return instance[-1]

ACTUAL_LABELS = [ instance_label(test_instance) for test_instance in TESTING_DATA ]

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
   list(filter(lambda row : row[-1] == LOW_QUALITY, data)),
   list(filter(lambda row : row[-1] == HIGH_QUALITY, data))
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

  #assert(len(low_rows) != 0)
  #assert(len(high_rows) != 0)
  #assert(instance_label(low_rows[0]) == LOW_QUALITY)
  #assert(instance_label(high_rows[0]) == HIGH_QUALITY)

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




def predict_with_knn(test_instance : list[float], k : int, train_data) -> int :


  instance_distance_to_label_array = [
      (
        distance_euclidean(instance_attributes(train_instance), test_instance),
        instance_label(train_instance)
      ) 
      for train_instance in train_data ]

  random.shuffle(instance_distance_to_label_array)

  most_similar_train_instances = sorted(
    instance_distance_to_label_array, # randmise order
    key = lambda x : x[0] # sort only by distance, not by label
  )

  labels = [label for (distance, label) in most_similar_train_instances]
  labels_of_k_closest = labels[:k]
  most_common_label_of_k_closest = Counter(labels_of_k_closest).most_common()[0][0]

  assert(
    (most_common_label_of_k_closest == HIGH_QUALITY) or 
    (most_common_label_of_k_closest == LOW_QUALITY) 
  )

  return most_common_label_of_k_closest


def check_accuracy(predict_instance_label : Callable[[list[float]], int], testing_data , prediction_name : str) :

  # test each instance in the test data
  predicted = [ predict_instance_label(test_instance) for test_instance in testing_data ]
  conf = confusion_matrix(ACTUAL_LABELS, predicted)


  accur = sum(conf[i][i] for i in range(len(conf))) / sum(sum(x) for x in conf)
  print("ACCURACY : " + str(accur) + " " + prediction_name)


  # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix


  plt.cla()
  plt.clf() # TODO avoid this


  plt.figure(figsize = (10,7))

  # on the side is actual
  # on the bottom is predicted

  sn.heatmap(conf, annot=True, fmt=".0f")
  plt.savefig("confusion/" + prediction_name)

  plt.cla()
  plt.clf()


  # False positive and False negatives 


  # TODO seperate false positive and false negatives



  # saying positive is high quality
  # false = 0, true = 1, negative = 0, positive = 0
  # [[falseNegative, falsePositive],
  #  [trueNegative,   truePositive]]

  trueFalseNegativePositive = [[[],[]],[[],[]]]
  for i, inst in enumerate(testing_data):

    isTrue = int(instance_label(inst) == predicted[i])
    isPositive = int(instance_label(inst))
    trueFalseNegativePositive[isTrue][isPositive].append(inst)


  flattened_confusion_columns = [trueFalseNegativePositive[0][0], trueFalseNegativePositive[0][1], trueFalseNegativePositive[1][0], trueFalseNegativePositive[1][1]] # TODO get order right

  
  for attrib_x_index, attrib_x in enumerate(ATTRIBUTES) : # tmp for testing
    for attrib_y_index, attrib_y in enumerate(ATTRIBUTES) :



      fig, ax = plt.subplots()

      # add in not high quality

      for columns, colour in zip(flattened_confusion_columns, ["green", "red", "blue", "black"]) : # TODO add labels

        xs = get_column(columns, attrib_x_index)
        ys =  get_column(columns, attrib_y_index) 

        ax.scatter(xs, ys, c=colour, alpha=0.3)
      
      plt.xlabel(attrib_x)
      plt.ylabel(attrib_y)

      plotTitle = "confusion_scatters/"+prediction_name+"/"+attrib_x+"_"+attrib_y
      plt.title(plotTitle)
      ax.grid(True)

      plt.savefig(plotTitle)

      plt.cla()
      plt.clf()


  # True and False


  






  plt.cla()
  plt.clf()

  return predicted


def min_max_scale(ls) :
  max_ls = max(ls)
  min_ls = min(ls)
  range_ls = max_ls - min_ls

  return [(x - min_ls) / range_ls for x in ls] 


assert(min_max_scale([0,10]) == [0,1])
assert(min_max_scale([5,10]) == [0,1])



def distribution_scale(ls : list) :
  mean = sum(ls) / len(ls)
  stddev = standardDeviation(ls)
  return [(x - mean) / stddev for x in ls]



def getColum(arr, c : int) :
  return [ row[c] for row in arr ]

def flip(arr : list[list[float]]) :

  return [getColum(arr, c) for c in range(len(arr[1]))]

def scaleColumns(matrix : list[list[float]], f) :

  matrix = flip(matrix)
  matrix = list(map(f, matrix[:-1])) + [list(matrix[-1])] # only scale attribute columns
  matrix = flip(matrix)

  return matrix


#assert(scaleColumns(np.array([[0,10],[5,5]]), min_max_scale) == [[0.0,1.0],[1.0,0.0]])



def mean(ls : list[float]) : 
  # TODO is this the right formula
  return sum(ls) / len(ls)


def variance(ls : list[float]) : 
  m = mean(ls)
  return sum([math.pow((x - m), 2) / len(ls) for x in ls])


def standardDeviation(ls : list[float]) :
  return math.sqrt(variance(ls))









####################################################################################################################################
# Code to run


# quick data summaries
"""
data_report(TRAINING_DATA)
data_report(TESTING_DATA)
"""

# scale columns
min_max_scaled_training_data = scaleColumns(TRAINING_DATA, min_max_scale)
min_max_scaled_test_data = scaleColumns(TESTING_DATA, min_max_scale)

distribution_scaled_training_data = scaleColumns(TRAINING_DATA, distribution_scale)
distribution_scaled_test_data = scaleColumns(TESTING_DATA, distribution_scale)

# generate graphs
"""
generate_all_distributions(TRAINING_DATA, "unscaled_training_data")
generate_all_distributions(min_max_scaled_training_data, "min_max_scaled_training_data")
generate_all_distributions(distribution_scaled_training_data, "distribution_scaled_training_data")
"""

# accuracy calculations
knn_predicted = \
  check_accuracy(lambda instnace : predict_with_knn(instnace, 1, TRAINING_DATA), TESTING_DATA, "knn") 
knn_with_min_max_normalisation_predicted = \
  check_accuracy(lambda instnace : predict_with_knn(instnace, 1, min_max_scaled_training_data), min_max_scaled_test_data, "knn_with_min_max_normalisation") 
knn_with_distribution_normalisation_predicted = \
  check_accuracy(lambda instnace : predict_with_knn(instnace, 1, distribution_scaled_training_data), distribution_scaled_test_data, "knn_with_distribution_normalisation") 




# TODO compare predicted



####################################################################################################################################


