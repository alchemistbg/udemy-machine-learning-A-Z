# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 19:11:38 2022

@author: Oggy
"""

# In order to read the csv file correctly, you need to set Spyder's
# working directory to point to the folder where the csv file is
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
dataset = pd.read_csv('./50_startups.csv')
# print(dataset)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# %%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)

# %%
y_test_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_test_pred.reshape(len(y_test_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# %%
import statsmodels.formula.api as smf
X_prime = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)