#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:18:11 2019

@author: aman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from pandas.plotting import scatter_matrix


dataset=pd.read_csv("/home/aman/ML/DataSet/housing.csv")
dataset.head()

X = dataset.iloc[:,[0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[:,8].values

#pd.plotting.scatter_matrix(dataset)


dataset.isnull().sum()


#dataset["total_bedrooms"].isnull().sum()




from sklearn.preprocessing import Imputer
im=Imputer(missing_values="NaN",strategy="median")
X[:,[4]]=im.fit_transform(X[:,[4]])

from sklearn.preprocessing import LabelEncoder
lib=LabelEncoder()
X[:,8]=lib.fit_transform(X[:,8])

from sklearn.preprocessing import OneHotEncoder
one =OneHotEncoder(categorical_features=[8])
X=one.fit_transform(X)
X = X.toarray()


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,y)
linreg.score(X,y)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X = sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_train,y_train)


ypred=lin.predict(X_test)

lin.score(X_test,y_test)


from 
#if radj is negetive then the model you created is not eliiglible for the prediction