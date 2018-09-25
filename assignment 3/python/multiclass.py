import os 
import pandas as pd
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
data = loadmat('ex3data1.mat')

def sigmoid(z):
    return 1/(1+np.exp(-z))
def cost(theta,X,y,learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
    reg = (learning_rate/(2*len(y)))*np.sum(np.power((theta[:,1:theta.shape[1]]),2))
    return np.sum(first-second)/(len(y)) + reg
def grad(theta,X,y,learning_rate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    error = (sigmoid(X*theta.T) - y)
    grad = ((X.T*error).T)/(len(y)) + (learning_rate/len(y))*theta 
    grad[0,0] = np.sum(np.multiply(error,X[:,0]))/(len(y));
    return np.array(grad).ravel()
def oneVsAll(X,y,num_labels,learning_rate):
    rows = X.shape[0]
    params = X.shape[1]
    all_theta = np.zeros((num_labels,(params+1)))
    X = np.insert(X,0, values= np.ones(rows),axis=1)
    for i in range(1,num_labels+1):
        theta = np.zeros(params+1)
        y_i = np.array([1 if label==i else 0 for label in y])
        y_i = np.reshape(y_i,(rows,1))
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=grad)
        all_theta[i-1,:] = fmin.x
    return all_theta
def predictAll(X,all_theta):
    rows = X.shape[0]
    X = np.insert(X,0,values=np.ones(rows),axis=1)
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    h = sigmoid(X*all_theta.T)
    h_arg = np.argmax(h,axis=1)
    h_arg = h_arg+1
    return h_arg
all_theta = oneVsAll(data['X'],data['y'],10,1)
pred = predictAll(data['X'],all_theta)
correct = [1 if a==b else 0 for (a,b) in zip(pred,data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))  