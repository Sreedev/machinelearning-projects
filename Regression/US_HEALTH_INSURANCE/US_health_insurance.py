#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:31:28 2020

@author: Sreedev

Model to predict the charges of insurance from BMI.
"""

# Importing the libraries
import pandas as pd
import numpy as np


'''IMPORTING DATASET'''
dataset = pd.read_csv('insurance.csv')
X = pd.DataFrame(dataset.iloc[:,:-1].values)
y = dataset.iloc[:, 6].values


'''DATA PRE-PROCESSING'''

#One hot encoded features with multiple unique categorial data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [5])], remainder='passthrough')
X = pd.DataFrame(ct.fit_transform(X))

# Label encoded feature with 2 unique categorial data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 5] = labelencoder_X.fit_transform(X.iloc[:, 5])
X.iloc[:, 8] = labelencoder_X.fit_transform(X.iloc[:, 8])


'''# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)'''


