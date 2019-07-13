#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:47:13 2019

@author: aman
"""

#importing the package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing my data set
data=pd.read_csv("/home/aman/Desktop/ML/Datasets/50_Startups.csv")


#creating a feature matrix
X=data.iloc[:,:-1].values

#creating the label vector
y = data.iloc[:,-1].values

#importing preprocessing a subpackage inside in sklearn library imprting
#class name as LabelEccoder which is used to convert categorical value to numbercal

from sklearn.preprocessing import LabelEncoder
#creating the object of class
lib=LabelEncoder()
#executing the functionality
X[:,3]=lib.fit_transform(X[:,3])


#import the class OneHotEncoder to remove the relation from a numerical categorical column

from sklearn.preprocessing import OneHotEncoder
#object of the class
one=OneHotEncoder()
X[:,3]=one.fit_transform(X[:,3])


#spliting the data set into training and testing model
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)


from sklearn.linear_model import LinearRegression
lig = LinearRegression()
lig.fit(X_train,y_train)

#creating prediction
y_pred= lig.predict(X_test)


#checking the accuracy of the model
lig.score(X_test,y_test)






