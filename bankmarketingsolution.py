#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 19:06:56 2019

@author: aman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("/home/aman/Desktop/ML/Datasets/bank-full.csv",delimiter=";")


X = dataset.iloc[:,[0,1,2,3,6]].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
lib=LabelEncoder()
X[:,1]=lib.fit_transform(X[:,1])
X[:,2]=lib.fit_transform(X[:,2])
X[:,3]=lib.fit_transform(X[:,3])
X[:,4]=lib.fit_transform(X[:,4])

y = lib.fit_transform(y)


from sklearn.preprocessing import OneHotEncoder
one=OneHotEncoder(categorical_features=[1,2,3,4])
X=one.fit_transform(X)
X=X.toarray()

from sklearn.preprocessing import StandardScaler
sim = StandardScaler()
X = sim.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
log  = LogisticRegression()
log.fit(X_train,y_train)








