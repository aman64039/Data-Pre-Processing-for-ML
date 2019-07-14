#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:52:37 2019

@author: aman
"""
#imorting the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("/home/aman/Desktop/ML/Datasets/Position_Salaries.csv")

#creating the feature of matrix
X = dataset.iloc[:,1].values
#label vector
y = dataset.iloc[:,2].values

#converting the array to matrix
X=X.reshape(-1,1)

#visulize the problem or data set
plt.scatter(X,y)
plt.show()

#using linear regression find the best fit line and again plot it and check the best fit line

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X,y)

#importing polynomial feature 
from sklearn.preprocessing import PolynomialFeatures
#passing the degree acc to the requirment
poly=PolynomialFeatures(degree=4)
#fitting the polynomial 
X_poly=poly.fit_transform(X)
poly.fit(X_poly,y)

#creating the another object of linear regression class
lin2=LinearRegression()
lin2.fit(X_poly,y)


#ploting the graph by using the polynomial fiunctio and get the best fit line
plt.scatter(X,y,color="red")
plt.plot(X,lin2.predict(X_poly),color="blue")
plt.show()




