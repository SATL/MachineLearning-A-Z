# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 18:54:14 2017

@author: eslem
"""
#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#split dataset to training and set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
 

#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#x_train = sc_x.fit_transform(x_train)
#x_test = sc_x.transform(x_test)
