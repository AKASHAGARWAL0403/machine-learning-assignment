import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import computeCost
import gradient

path = os.getcwd()+'\ex1data1.txt'
data = pd.read_csv(path,header=None , names=['Population','Profit'])

X = data.iloc[:,0:1]
y = data.iloc[:,1:2]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             

X  = np.matrix(X.values)
y  = np.matrix(y.values)

X = np.insert(X,0,1.0,axis = 1)

theta = np.zeros((2,1))

cost = computeCost.computeCost(X,y,theta)
print(cost)

[theta,cost] = gradient.gradient(X,y,theta,.01,1000)
print(theta)
