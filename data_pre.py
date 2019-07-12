#Importing the package
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#Get the Data bu using pandas read method
dataset = pd.read_csv('dataset/Data_Pre.csv')

#X is feature of matrix
#slicing
X = dataset.iloc[:, 0:3].values

#Y is vector of Label
#slicing
y = dataset.iloc[:, -1].values


#fom sklearn package there is a subpackage name as impute and inside this there is a class name as SimpleImputer.
#SimpleImputer is used to fill the nan values by mean value taht is the default we can also chnge the startergy into median as well by using statergy parameter in SimputerImputer Class
from sklearn.impute import SimpleImputer
#creating the object of SimpleImputer class
sim = SimpleImputer()
#Fitting the SimpleImputer
sim.fit(X[:, 0:2])
#Transform the chnaged 
X[:, 0:2] = sim.transform(X[:, 0:2])


#using preprocessing sub package which is inside the sklearn package and importing the class name as LabelEncoder which is used to convert categorical value to numerical coulmn mean give a unique number to each categorical value because our ML works on Digits
from sklearn.preprocessing import LabelEncoder
#creating object of LabelEncoder Class
lab = LabelEncoder()
#Fitting and Transforming
X[:, 2] = lab.fit_transform(X[:, 2])
#lab.classes_ is used to check digit value of each categorical values
lab.classes_
y = lab.fit_transform(y)


#using preprocessing sub package which is inside the sklearn package and importing the class name as OneHotEncoder which is used to give the number variable which is created by LabelEncoer into one range like if i have a and b after labelencoding it will give numbe rlike 1 and 2 so this onehotencoder will remove the relation like 2>1 these all number will be in one range
from sklearn.preprocessing import OneHotEncoder
#creating object of a class
one = OneHotEncoder(categorical_features = [2])
X = one.fit_transform(X)
X = X.toarray()


"""using preprocessing sub package which is inside the sklearn package and importing the class name as StandardScaler which is used to convert all the column at one scale.like if i am giving an example 
one person is running with the speed of 4km/h
second is running 4000m/h
third one is running 400m/sec
so we have to check whose speed is best we cant find that because all are in diffrent unit 
so first we have to make them as a same unit so in the same manner here Standard Scaler is used to make the column at same sale
"""
from sklearn.preprocessing import StandardScaler
#creating object of This Class
sc = StandardScaler()
X = sc.fit_transform(X)










