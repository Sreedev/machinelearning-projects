#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 18:31:28 2020

@author: Sreedev

Model to predict the charges of insurance from BMI.
"""

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, 2].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2, random_state = 0)


X_train = X_train.reshape(1,-1) 
y_train = y_train.reshape(1,-1) 
X_test = X_test.reshape(1,-1) 
y_test = y_test.reshape(1,-1)

"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('BMI vs Insurance charges (Test set)')
plt.xlabel('BMI')
plt.ylabel('Insurance charge')
plt.show()