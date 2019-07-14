#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:08:02 2019

@author: aman
"""

#import some packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#m is number observation
m = 100

#X will be features of matrix
X=6*np.random.randn(m,1)

#y will be your polynomail equation
#y = b0+b1x+b2x**2
y = 2+ 1*X + 0.05 * X**2 + np.random.randn(m,1)
 

#ploting a scatter graph for the data set
plt.scatter(X,y)
#this is the axis i want -10 to 10 is for X axis and 0 to 9 for y axis
plt.axis([-10,10,0,9])
plt.show()

#importing the poly
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)


X_new=np.linspace(-3,3,100).reshape(-1,1)
X_new_poly=poly.fit_transform(X_new)

y_new=lin.predict(X_new)

plt.scatter(X,y)
#this is the axis i want -10 to 10 is for X axis and 0 to 9 for y axis
plt.axis([-10,10,0,9])
plt.plot(X_new_poly,y_new,c="r")
plt.show()




#importing the Linear Regression class for getting the best fit line
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X,y)

y_pred=lin.predict(X)


plt.scatter(X,y)
#this is the axis i want -10 to 10 is for X axis and 0 to 9 for y axis
plt.axis([-10,10,0,9])
plt.plot(X,y_pred)
plt.show()
