#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 12:02:30 2019

@author: aman
"""

"""
Logistic regression produce a result in binary format which 
is used to predict the outcome of caategorical dependdent 
variable. So the output should be discreat or categorical
classification family of algo are mostly of two type
probablistic and determinstic
logistic regression bellong to probabilstic family of
classifiction
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.arange(-10,10,0.01)

sig = 1/(1+np.power(np.e,-x))


plt.plot(x,sig)
plt.show()

#y = mx+c
lin= 3*x+7

plt.plot(x,lin)
plt.show()











