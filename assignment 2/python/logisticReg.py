import os
import numpy as np
import pandas as pd

path = os.getcwd()+"\ex2data1.txt";
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])  
data.head()  

data.insert(0,'Ones',1)

col = data.shape[1]

X = data.iloc[:,0:col-1]  
Y = data.iloc[:,col-1:col]

X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta = np.zeros((3,1))

def sigmoid(z):
    return 1/(np.exp(-z)+1)
def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
       # print(term.shape)
        grad[i] = np.sum(term) / len(X)

    return grad
import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y)) 
cost(result[0],X,Y)