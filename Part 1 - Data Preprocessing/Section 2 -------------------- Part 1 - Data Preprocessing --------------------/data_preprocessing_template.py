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

