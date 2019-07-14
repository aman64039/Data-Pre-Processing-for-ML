#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:38:13 2019

@author: aman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



m = 100

X = 6* np.random.randn(m,1)
y = 2+X+0.05*X**2 + np.random.randn(m,1)
 


plt.scatter(X,y)
plt.show()
 

from sklearn.preprocessing import PolynomialFeatures
poly= PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_poly,y)


X_new=np.linspace(-10,10,100).reshape(-1,1)
X_new_poly=poly.fit_transform(X_new)

y_new=lin.predict(X_new_poly)

plt.scatter(X,y)
plt.plot(X_new,y_new,c="r")
plt.show()
