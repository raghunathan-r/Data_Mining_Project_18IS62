# Program to Binarize values form the given dataset.

# importing the required libraries
# numpy -> for reading the csv files and reshaping arrays
# matplotlib.pyplot -> for
# pandas -> for extracting rows using 'iloc'

    # make sure you have numpy installed -> pip3 install numpy | pip3 install matplotlib | pip install pandas | pip install sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the Sklearn library
from sklearn import preprocessing

# importing the dataset form a .csv file
file_path = './Sample_Salaries_Data_Set.csv'
data_set = pd.read_csv(file_path)
    # .head() is used to selecting a number of rows. (n) means n rows will be selected.
data_set.head()

print("\nThe data set being used is :\n\n", data_set, "\n\n")

# extracting the values of different columns using slicing
    # .iloc or .loc is used to extract the data form rows. () will extract from all rows. (n) will extract from the 'n' row.
    # .iloc[:, n] selects the column until the nth column but excluding the nth column.
age = data_set.iloc[:, 1].values
salary = data_set.iloc[:, 2].values

# printing all the extracted values to check if its right
print("\nThe extracted age values : \n", age)
print("\nThe extracted salary values : \n", salary)

# now binarizing the values

from sklearn.preprocessing import Binarizer

    # reshape from numpy is used to convert a 2D array into a 1D array or vice versa
    # here reshape(1, -1) is used as '1' -> means single feature is there in date. '-1' -> to reshape the 2D array into a 1D array
    # reference : https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
x = age
x = x.reshape(1, -1)
y = salary
y = y.reshape(1, -1)

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
    # Values greater than the threshold map to 1, while values less than or equal to the threshold map to 0. With the default threshold of 0, only positive values map to 1.
    # Binarizer(35) means it will map all values greater than 35 to 1
binarizer_1 = Binarizer(threshold = 35)
binarizer_2 = Binarizer(threshold = 61000)

# transforming the features / columns
    # fit_transform(x) fits and transforms the date and returns the transformed version of x
print("\nBinarized age [threshold - 35] : \n", binarizer_1.fit_transform(x))
print("\nBinarized salary [threshold - 61000]: \n", binarizer_2.fit_transform(y))

# 
# trying label Binarizer

from sklearn.preprocessing import LabelBinarizer
lable_bin = LabelBinarizer()


# countries = data_set.iloc[:, 0].values
# z = countries
# z = z.reshape(1, -1)

print("\nBinarized countries [France - 0]: \n", lable_bin.fit_transform(data_set['Country']))