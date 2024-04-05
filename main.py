
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import csv


# import data
# https://diveintopython.org/learn/file-handling/csv


with open('COMP30027_2024_asst1_data/winequality-test.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

        

# KNN 
