import os
import numpy as np
import pandas as pd
import computeCost
import gradientMulti

path = os.getcwd()+'\ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])  
data2.head()  

data2 = (data2- data2.mean())/data2.std()
data2.head()
data2.insert(0,'Ones',1)

col = data2.shape[1]

X = data2.iloc[:,0:col-1]  
Y = data2.iloc[:,col-1:col]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.zeros((3,1))
cost  = computeCost.computeCost(X,Y,theta)

theta,cost2 = gradientMulti.gradient(X,Y,theta,.01,1000)
cost  = computeCost.computeCost(X,Y,theta)
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(1000), cost2, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Eoch')  
