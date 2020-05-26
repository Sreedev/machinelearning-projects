#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:31:28 2020

@author: Sreedev

Model to predict the charges of insurance from BMI.
"""

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = pd.DataFrame(dataset.iloc[:,:-1].values)
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

'''#Reshaping the dataset since it has only once dimention
X_train = X_train.reshape(1,-1) 
y_train = y_train.reshape(1,-1) 
X_test = X_test.reshape(1,-1) 
y_test = y_test.reshape(1,-1)'''

#Encoding categorical data


