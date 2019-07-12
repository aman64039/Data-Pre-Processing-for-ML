#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:10:03 2019

@author: aman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("/home/aman/Desktop/ML/suv.csv")
data

X = data.iloc[:,[2,3]].values
y = data.iloc[:,4]


from sklearn.preprocessing import StandardScaler
#from sklearn.impute import SimpleImputer
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train , X_test,y_train,y_test=train_test_split(X,y,test_size =0.2)


from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)

y_pred = log.predict(X_test)

log.score(X_test,y_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_pred,y_test)



